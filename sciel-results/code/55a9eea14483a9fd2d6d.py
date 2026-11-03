"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def _compute_radius(pts):
    """Compute the max common radius for given centers."""
    n = len(pts)
    # Distances to boundary
    dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(dists)
    # Pairwise distances using Delaunay for efficiency if n is large, but for exact min we need all pairs
    # For n=300, O(n^2) is fine (90k pairs)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair = np.min(dist_matrix) / 2.0
    return min(min_boundary, min_pair)

def _get_delaunay_edges(pts):
    """Get edges of the Delaunay triangulation."""
    try:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    a, b = simplex[i], simplex[j]
                    edges.add((min(a, b), max(a, b)))
        return list(edges)
    except Exception:
        return []

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s
    
    # Step 1: Hexagonal initialization
    # Generate a dense hexagonal grid and pick the best N
    spacing = 0.1
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        dist = np.hypot(points[:, 0], points[:, 1])
        # Keep points that can fit a circle of radius spacing/2 inside unit circle
        mask = dist <= 1.0 - spacing / 2.0
        points = points[mask]
        dist = dist[mask]
        if len(points) >= N:
            idx = np.argsort(dist, kind='stable')[:N]
            pts = points[idx].copy()
            break
        spacing *= 0.95
    
    best_radius = _compute_radius(pts)
    best_pts = pts.copy()
    
    if time.time() > deadline - 5.0:
        return best_pts
    
    # Step 2: Local search using Delaunay edges
    # We will perform iterative local moves to increase the minimum distance.
    # Strategy: For each point, try to move it in the direction away from its closest neighbors.
    # Or more simply: Coordinate descent / random perturbation.
    
    # A robust approach: 
    # 1. Compute Delaunay graph.
    # 2. For each point, calculate the "pressure" from neighbors.
    # 3. Move points slightly in the direction of least pressure (or away from closest neighbor).
    
    # Let's use a simple gradient-free local optimization:
    # For each point, calculate the closest neighbor distance.
    # If moving a point increases the minimum distance in its local neighborhood, do it.
    
    # Better: Use the fact that at optimum, points are often in equilibrium of forces from neighbors.
    # We can simulate this by applying repulsive forces.
    
    # Let's implement a force-directed layout relaxation.
    # Force on point i is sum of repulsive forces from neighbors j.
    # F_ij = k * (p_i - p_j) / |p_i - p_j|^3 (Coulomb-like)
    # Plus a restoring force to stay inside? No, just boundary constraint.
    
    # Actually, a standard technique for circle packing is to maximize the minimum distance.
    # We can use a simple iterative scheme:
    # 1. Find closest pair.
    # 2. Move the two points slightly apart.
    # 3. Check boundary.
    
    # Let's try a more robust method: 
    # Randomly perturb points and accept if radius improves. (Simulated Annealing style)
    
    current_pts = best_pts.copy()
    current_radius = best_radius
    
    step_size = 0.01
    n_iter = 0
    max_iter = 50000
    
    while time.time() < deadline - 10.0 and n_iter < max_iter:
        n_iter += 1
        
        # Pick a random point
        i = np.random.randint(N)
        orig_pos = current_pts[i].copy()
        
        # Calculate current closest distance for point i
        dists = np.sqrt(np.sum((current_pts - current_pts[i])**2, axis=1))
        dists[i] = np.inf
        min_dist = np.min(dists)
        
        # Try moving in a random direction
        angle = np.random.uniform(0, 2*np.pi)
        new_pos = current_pts[i] + step_size * np.array([np.cos(angle), np.sin(angle)])
        
        # Check boundary
        if np.linalg.norm(new_pos) > 1.0 - 1e-9:
            # Project back to circle or skip
            # Let's just skip if out of bounds to keep it simple and valid
            continue
        
        current_pts[i] = new_pos
        
        # Check if radius improved
        # We only need to check distances involving point i and the global min
        # But computing full radius is O(N^2). 
        # Optimization: Check only distances from i to others, and the previous global min.
        new_dists_i = np.sqrt(np.sum((current_pts - current_pts[i])**2, axis=1))
        new_dists_i[i] = np.inf
        local_min = np.min(new_dists_i)
        
        # The global min is min(local_min, other_pairs_min)
        # We don't have other_pairs_min easily without recomputing.
        # For N=300, O(N^2) is ~90k ops. 50k iters * 90k ops = 4.5e9 ops. Too slow in Python.
        
        # We need a faster check.
        # Let's stick to the force-directed approach which can be vectorized or at least more efficient if we only update neighbors.
        
        # Let's restart with a simpler, vectorized gradient step on the objective:
        # f(x) = min( min_i (1-|x_i|), min_{i<j} |x_i-x_j|/2 )
        # We can approximate the gradient of the min by using only the active constraints (those achieving the min).
        
        # Let's compute the active constraints for the current best_pts
        # 1. Boundary active: points where 1-|x_i| is close to min_boundary
        # 2. Pair active: pairs where |x_i-x_j|/2 is close to min_pair
        
        # This is getting complex. Let's use a standard library optimizer on a subset or with a simpler objective.
        
        # Alternative: Use scipy.optimize.minimize on the objective -min_dist, but with a simpler initialization and fewer iterations.
        # The previous attempt failed due to a bug. Let's fix the bug and use L-BFGS-B.
        
        # The bug was: `grad[mask_b] += weights[mask_b][:, np.newaxis] * ...`
        # `weights` has size N + N(N-1)/2? No, `all_vals` has size N + N(N-1)/2? 
        # In the previous code: `all_vals = np.concatenate([dists_boundary, half_dists.ravel()])`
        # `half_dists` is N x N. `ravel()` makes it N^2.
        # So `all_vals` has size N + N^2.
        # But the pairwise distances are symmetric, so we are double counting.
        # Also, the index calculation `weights[n + i*n + j]` assumes a flattened N x N matrix.
        # The error "size of axis is 90300" (which is 300*301) vs "300" suggests a mismatch in indexing.
        
        # Let's rewrite the gradient calculation correctly.
        
        break # Exit the while loop to use the optimizer below
    
    # Let's use L-BFGS-B with a corrected gradient.
    
    def _obj_and_grad(x, beta=50.0):
        """
        Objective: -soft_min(constraints)
        Constraints:
        1. Boundary: c_i = 1 - |x_i|
        2. Pairwise: d_ij = |x_i - x_j| / 2 for i < j
        """
        p = x.reshape(-1, 2)
        n = p.shape[0]
        
        # Boundary distances
        dists_b = 1.0 - np.hypot(p[:, 0], p[:, 1])
        
        # Pairwise distances (only i < j to avoid duplication and diagonal)
        # We can compute full matrix and take upper triangle
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        # Mask out diagonal
        idx = np.triu_indices(n, k=1)
        dists_p = dist_matrix[idx] / 2.0
        
        all_vals = np.concatenate([dists_b, dists_p])
        
        # Soft min
        m = np.min(all_vals)
        exp_vals = np.exp(-beta * (all_vals - m))
        weights = exp_vals / np.sum(exp_vals)
        
        # Objective value
        obj = -(m - np.log(np.sum(exp_vals)) / beta)
        
        # Gradient
        grad = np.zeros_like(p)
        
        # Boundary part
        # d/dx_i (1 - |x_i|) = -x_i / |x_i|
        r = np.hypot(p[:, 0], p[:, 1])
        mask_b = r > 1e-12
        # weights[:n] corresponds to dists_b
        grad[mask_b] -= weights[:n][mask_b][:, np.newaxis] * p[mask_b] / r[mask_b][:, np.newaxis]
        
        # Pairwise part
        # d/dx_i (|x_i - x_j| / 2) = 0.5 * (x_i - x_j) / |x_i - x_j|
        # d/dx_j (|x_i - x_j| / 2) = 0.5 * (x_j - x_i) / |x_i - x_j|
        
        # We have weights for the pairs in `dists_p` which corresponds to upper triangle indices
        # We need to accumulate gradients for each point.
        
        # To be efficient, we can loop over pairs if N is small, but N=300 is 45k pairs.
        # Looping 45k times in Python is slow but maybe acceptable for a few iterations?
        # Better: Vectorize using the indices.
        
        # Create a matrix of weights for pairs, 0 on diagonal and lower triangle
        w_matrix = np.zeros((n, n))
        w_matrix[idx] = weights[n:] # weights for pairs start at index n
        
        # Compute direction matrix
        # dir[i,j] = (p_i - p_j) / |p_i - p_j|
        d_safe = np.where(dist_matrix > 1e-12, dist_matrix, 1.0)
        direction_matrix = diff / d_safe[:, :, np.newaxis]
        # Zero out where dist is 0
        direction_matrix = np.where(dist_matrix[:, :, np.newaxis] > 1e-12, direction_matrix, 0.0)
        
        # grad[i] += sum_j 0.5 * w_matrix[i,j] * direction_matrix[i,j]
        # Note: w_matrix is symmetric? No, we only filled upper triangle.
        # But the force is symmetric. 
        # Let's fill the full matrix to use matrix multiplication.
        w_full = w_matrix + w_matrix.T
        # But we must be careful not to double count if we sum over all j.
        # If we sum over all j, we get 0.5 * (w_ij * dir_ij + w_ji * dir_ji).
        # Since w_ij = w_ji (if we symmetrize) and dir_ji = -dir_ij.
        # Wait, the derivative of dist_ij wrt x_i is 0.5 * dir_ij.
        # The derivative of dist_ij wrt x_j is 0.5 * dir_ji = -0.5 * dir_ij.
        # In our objective, we have term dist_ij.
        # So grad_x_i gets contribution 0.5 * w_ij * dir_ij.
        # grad_x_j gets contribution 0.5 * w_ij * dir_ji.
        
        # If we define w_full[i,j] = w_ij for i!=j, then:
        # grad[i] = sum_j 0.5 * w_full[i,j] * direction_matrix[i,j]
        
        grad += 0.5 * np.sum(w_full[:, :, np.newaxis] * direction_matrix, axis=1)
        
        return obj, grad.ravel()
    
    x0 = best_pts.ravel()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    from scipy.optimize import minimize
    
    # Use L-BFGS-B
    # We need to be careful with the time limit.
    # L-BFGS-B can be slow.
    
    try:
        result = minimize(
            lambda x: _obj_and_grad(x)[0], 
            x0, 
            method='L-BFGS-B',
            jac=lambda x: _obj_and_grad(x)[1],
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20}
        )
        
        if time.time() < deadline - 5.0:
            candidate = result.x.reshape(-1, 2)
            # Ensure validity
            dists = np.hypot(candidate[:, 0], candidate[:, 1])
            if np.any(dists > 1.0):
                mask_out = dists > 1.0
                candidate[mask_out] /= dists[mask_out][:, np.newaxis]
            
            r = _compute_radius(candidate)
            if r > best_radius:
                best_pts = candidate
                best_radius = r
    except Exception:
        pass
    
    # Final validity check
    final_pts = best_pts.copy()
    dists = np.hypot(final_pts[:, 0], final_pts[:, 1])
    if np.any(dists > 1.0):
        mask_out = dists > 1.0
        final_pts[mask_out] /= dists[mask_out][:, np.newaxis]
    
    return final_pts
# EVOLVE-BLOCK-END
