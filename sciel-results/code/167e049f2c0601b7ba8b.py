import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
_DEADLINE_MARGIN = 10.0 # Seconds before timeout to stop

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_radius_and_worst(pts):
    """
    Compute the exact covering radius and identify the worst facet.
    Returns:
        radius: float, the maximum angular circumradius in radians.
        worst_facet_indices: np.array of shape (3,) or None if hull fails.
        worst_pole: np.array of shape (3,) or None.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        if M == 0:
            return np.pi, None, None

        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi, None, None
        
        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        # Angular circumradius for each valid facet
        # cos(theta) = normal . vertex
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)
        
        max_radius = np.max(ang_radii)
        worst_local_idx = np.argmax(ang_radii)
        
        # Map back to original face indices
        valid_indices = np.where(valid)[0]
        worst_face_idx = valid_indices[worst_local_idx]
        worst_facet_indices = faces[worst_face_idx]
        worst_pole = normals[worst_local_idx]
        
        return max_radius, worst_facet_indices, worst_pole
    except Exception:
        return np.pi, None, None

def _soft_hull_objective_and_grad(pts, grid, beta):
    """
    Compute a smooth approximation of the covering radius using log-sum-exp.
    Objective: (1/beta) * log(sum(exp(beta * d_i^2)))
    where d_i is the angular distance from grid point i to its nearest point in pts.
    
    Gradient is computed analytically.
    """
    # pts: (N, 3)
    # grid: (G, 3)
    
    # Compute squared angular distances.
    # cos(d) = dot(g, p). d = arccos(dot).
    # For small d, d^2 ~ (1 - cos(d)) * 2.
    # To avoid issues with arccos near 1, we can optimize based on squared chord length or similar,
    # but angular distance is the metric.
    # Let's use d^2 directly for the objective to emphasize the max.
    
    # Dot products: (G, N)
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angular distances: (G, N)
    dists = np.arccos(dots)
    dists_sq = dists ** 2
    
    # Find min distance for each grid point (max dot product)
    # min_dists_sq: (G,)
    min_dists_sq = np.min(dists_sq, axis=1)
    
    # Soft max approximation of the max of min_dists_sq
    # f(x) = (1/beta) log(sum(exp(beta * x)))
    # We want to minimize the max, so we minimize this soft max.
    
    # Numerical stability: subtract max
    max_val = np.max(min_dists_sq)
    exp_terms = np.exp(beta * (min_dists_sq - max_val))
    log_sum = np.log(np.sum(exp_terms))
    obj_val = (log_sum + beta * max_val) / beta
    
    # Gradient w.r.t pts
    # d(obj)/d(pts_j) = sum_i ( d(obj)/d(min_dists_sq_i) * d(min_dists_sq_i)/d(pts_j) )
    # 
    # Let w_i = exp(beta * (min_dists_sq_i - max_val)) / sum(exp(...))
    # d(obj)/d(min_dists_sq_i) = w_i
    #
    # min_dists_sq_i = min_k d_i(k)^2
    # Let k_i be the index of the nearest point for grid point i.
    # d(min_dists_sq_i)/d(pts_j) = 0 if j != k_i
    # if j == k_i: d(d_i^2)/d(p_j)
    #
    # d_i = arccos(dot(g_i, p_j))
    # d(d_i)/d(p_j) = -1/sqrt(1-dot^2) * g_i
    # d(d_i^2)/d(p_j) = 2 * d_i * (-g_i / sqrt(1-dot^2))
    
    # Identify nearest neighbor for each grid point
    nearest_idx = np.argmin(dists_sq, axis=1)
    
    # We need to accumulate gradients for each point j.
    # grad[j] += sum_{i: nearest_idx[i]==j} w_i * 2 * d_i * (-g_i / sqrt(1-dot^2))
    
    # Precompute terms
    # dots for nearest neighbors
    dots_nearest = dots[np.arange(len(grid)), nearest_idx]
    dists_nearest = dists[np.arange(len(grid)), nearest_idx]
    
    # sqrt(1 - dot^2)
    sin_dists = np.sqrt(np.clip(1.0 - dots_nearest**2, _EPS, None))
    
    # d(d^2)/d(p) = -2 * d / sin(d) * g
    # Note: this is the derivative of the distance squared w.r.t the vector p.
    # Since p is on the sphere, we must project the gradient to the tangent space.
    # grad_tangent = grad_euclidean - (grad_euclidean . p) * p
    
    weights = exp_terms / np.sum(exp_terms)
    
    # Vectorized accumulation
    # grad_euc[j] = sum_{i: k_i=j} weights[i] * (-2 * dists_nearest[i] / sin_dists[i]) * grid[i]
    
    grad_euc = np.zeros_like(pts)
    # Use bincount or scatter
    # coeff[i] = weights[i] * (-2 * dists_nearest[i] / sin_dists[i])
    coeffs = weights * (-2.0 * dists_nearest / sin_dists)
    
    # grad_euc[j] = sum_{i} coeffs[i] * grid[i] * (nearest_idx[i] == j)
    # This is equivalent to:
    # grad_euc = (coeffs[:, None] * grid).T @ one_hot(nearest_idx)
    # Or simply:
    for j in range(N):
        mask = (nearest_idx == j)
        if np.any(mask):
            grad_euc[j] = np.sum(coeffs[mask, None] * grid[mask], axis=0)
    
    # Project to tangent space
    # grad_proj[j] = grad_euc[j] - (grad_euc[j] . pts[j]) * pts[j]
    dot_grads = np.sum(grad_euc * pts, axis=1, keepdims=True)
    grad_proj = grad_euc - dot_grads * pts
    
    return obj_val, grad_proj

def _local_polish(pts):
    """
    Use the convex hull method to polish the solution.
    This handles the "exact" metric and fixes any smoothness artifacts.
    """
    current_radius, worst_facet, worst_pole = _compute_hull_radius_and_worst(pts)
    for _ in range(50):
        if worst_facet is None:
            break
        # Try to push the worst facet
        # Use a small step
        step = 0.05
        cand = pts.copy()
        for v in worst_facet:
            p = pts[v]
            dir_vec = worst_pole - p
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm > _EPS:
                new_p = p + step * dir_vec
                new_p = new_p / (np.linalg.norm(new_p) + _EPS)
                cand[v] = new_p
        
        cand_radius, new_worst_facet, new_worst_pole = _compute_hull_radius_and_worst(cand)
        if cand_radius < current_radius:
            pts = cand
            current_radius = cand_radius
            worst_facet = new_worst_facet
            worst_pole = new_worst_pole
        else:
            # Try smaller step
            step = 0.01
            cand = pts.copy()
            for v in worst_facet:
                p = pts[v]
                dir_vec = worst_pole - p
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > _EPS:
                    new_p = p + step * dir_vec
                    new_p = new_p / (np.linalg.norm(new_p) + _EPS)
                    cand[v] = new_p
            cand_radius, new_worst_facet, new_worst_pole = _compute_hull_radius_and_worst(cand)
            if cand_radius < current_radius:
                pts = cand
                current_radius = cand_radius
                worst_facet = new_worst_facet
                worst_pole = new_worst_pole
            else:
                break
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - _DEADLINE_MARGIN
    
    # Initial configuration
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius, _, _ = _compute_hull_radius_and_worst(pts)
    
    # Generate a grid for the soft hull objective
    # We need a grid dense enough to approximate the max well, but not too dense for speed.
    # N=130. Covering radius ~ 11 deg.
    # Grid resolution: 1 deg is ~ 360*180 = 64800 points.
    # Let's use a Fibonacci grid for the evaluation points too, maybe 5000 points.
    G = 5000
    i_grid = np.arange(G) + 0.5
    z_grid = 1.0 - 2.0 * i_grid / G
    phi_grid = np.pi * (1.0 + np.sqrt(5.0)) * i_grid
    r_grid = np.sqrt(np.clip(1.0 - z_grid * z_grid, 0.0, None))
    grid = np.column_stack([r_grid * np.cos(phi_grid), r_grid * np.sin(phi_grid), z_grid])
    
    # Riemannian Gradient Descent
    # Parameters
    lr = 0.1
    beta = 10.0 # Temperature for soft max. Higher is sharper (closer to max).
    
    # Adaptive Beta: Start low (smooth), increase to sharpen.
    # Or keep it fixed.
    
    while time.time() < deadline - 20.0: # Leave 20s for polish
        obj_val, grad = _soft_hull_objective_and_grad(pts, grid, beta)
        
        # Normalize gradient to control step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm > _EPS:
            grad = grad / grad_norm * lr
        
        # Update
        new_pts = pts + grad
        # Project back to sphere
        new_pts = new_pts / (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        # Check if this improves the *exact* hull radius (or a proxy)
        # Using exact hull radius every step is expensive.
        # Let's check every 10 steps.
        # For now, accept if obj_val decreases (approximation)
        # But we must ensure we don't diverge.
        
        # Simple acceptance: always accept for GD, but monitor exact radius periodically.
        pts = new_pts
        
        # Periodic check of exact radius
        if np.random.random() < 0.1: # 10% of steps
            curr_radius, _, _ = _compute_hull_radius_and_worst(pts)
            if curr_radius < best_radius:
                best_radius = curr_radius
                best_pts = pts.copy()
                
            # If we got worse than best, maybe revert?
            # Or just keep going, GD might escape.
            
        # Adjust beta?
        # Let's increase beta slowly to sharpen the objective
        beta *= 1.01
        if beta > 100.0:
            beta = 100.0
            
        # Decrease learning rate slowly
        lr *= 0.995
        if lr < 0.001:
            lr = 0.001

    # Final Polish
    final_pts = _local_polish(best_pts)
    
    # Check if polish improved
    final_radius, _, _ = _compute_hull_radius_and_worst(final_pts)
    if final_radius < best_radius:
        best_pts = final_pts
        best_radius = final_radius
        
    # Final normalization
    final_pts = np.array(best_pts, dtype=float)
    final_pts /= (np.linalg.norm(final_pts, axis=1, keepdims=True) + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(final_pts)) or final_pts.shape != (N, 3):
        return _initial_fib()
        
    return final_pts

if __name__ == "__main__":
    pts = construct()
    r, _, _ = _compute_hull_radius_and_worst(pts)
    print(f"Covering radius: {np.degrees(r)} degrees")
