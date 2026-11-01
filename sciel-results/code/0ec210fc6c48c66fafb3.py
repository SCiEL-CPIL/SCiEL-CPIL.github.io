import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12
_DEADLINE = 145.0  # Safety margin before 150s kill

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _generate_grid(n=4000):
    """
    Generate a dense, uniform grid of points on the sphere for the surrogate objective.
    Using a Fibonacci lattice ensures good coverage without clustering.
    """
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_radius(pts):
    """
    Compute the exact covering radius using the convex hull.
    Returns the maximum angular distance from the sphere to the nearest point.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[faces[:, 2]]] # Typo in original? No, faces[:, 2]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi
        
        normals = cross[valid].copy()
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return radii.max()
    except Exception:
        return np.pi

def _surrogate_objective_and_grad(flat_pts, grid):
    """
    Compute the surrogate objective: mean squared angular distance to nearest neighbor.
    And its gradient with respect to the flat point coordinates.
    
    Objective: J = (1/M) * sum_{g in grid} (d(g, NN(g)))^2
    where d is angular distance.
    
    Gradient derivation:
    Let g be a grid point, p_i be the nearest point to g.
    d = arccos(g . p_i)
    J = sum d^2
    dJ/dp_i = 2 * d * (dd/dp_i)
    
    d = arccos(c), c = g . p_i
    dd/dc = -1/sqrt(1-c^2)
    dc/dp_i = g
    
    dd/dp_i = -g / sqrt(1-c^2)
    
    However, p_i is constrained to the sphere. We compute the gradient in R^3,
    and the optimizer (L-BFGS-B) doesn't know about the sphere constraint directly
    unless we parameterize. 
    Alternative: Use Cartesian coordinates but project back to sphere after each step?
    No, standard minimize works on unconstrained or box-constrained.
    
    Better approach for differentiability on sphere:
    Use a parameterization? Or just use Cartesian and add a penalty for norm != 1?
    Or, simpler: Use the gradient in the tangent space?
    
    Let's stick to Cartesian coordinates for the optimizer, but the objective function
    will implicitly assume points are on the sphere. If they drift, the objective increases.
    To handle the constraint properly, we can use the fact that the gradient of the
    objective with respect to the Cartesian coordinates, projected onto the tangent plane,
    is the true gradient on the manifold.
    
    However, `scipy.optimize.minimize` with L-BFGS-B expects gradients in the coordinate system
    it operates in. If we operate in R^3, the gradient should point in the direction of
    steepest descent in R^3.
    
    Let's calculate the gradient in R^3.
    For a grid point g, let i be the index of the nearest point p_i.
    The contribution to the objective is d^2.
    grad_p_i (d^2) = 2 * d * grad_p_i (d)
    grad_p_i (d) = -g / sin(d)  (since sin(d) = sqrt(1-c^2))
    
    So grad_p_i = -2 * d / sin(d) * g
    
    If we sum over all grid points, for each point p_k, the gradient is the sum of contributions
    from all grid points for which p_k is the nearest neighbor.
    
    This is vectorizable.
    """
    pts = flat_pts.reshape(N, 3)
    
    # Compute distances from all grid points to all candidate points
    # dist_matrix[i, j] = angular distance between grid[i] and pts[j]
    dots = np.dot(grid, pts.T) # shape (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots) # shape (M, N)
    
    # Find nearest neighbor for each grid point
    nn_indices = np.argmin(dists, axis=1) # shape (M,)
    min_dists = dists[np.arange(len(grid)), nn_indices] # shape (M,)
    
    # Objective: mean squared distance
    obj = np.mean(min_dists ** 2)
    
    # Gradient
    # For each grid point m, it contributes to the gradient of pts[nn_indices[m]]
    # Contribution: -2 * d_m / sin(d_m) * g_m
    
    # We need to accumulate these contributions.
    # grad_pts shape (N, 3)
    grad_pts = np.zeros_like(pts)
    
    # Vectorized accumulation
    # weights[m] = -2 * d_m / sin(d_m)
    # Note: if d_m is very small, sin(d_m) is small, weight is large.
    # If d_m is 0, weight is 0/0. Limit is -2.
    sin_dists = np.sin(min_dists)
    # Handle division by zero
    valid_mask = sin_dists > _EPS
    weights = np.zeros_like(min_dists)
    weights[valid_mask] = -2.0 * min_dists[valid_mask] / sin_dists[valid_mask]
    weights[~valid_mask] = -2.0 # Limit as d->0 of -2d/sin(d) is -2
    
    # Accumulate: grad_pts[nn_indices[m]] += weights[m] * grid[m]
    # Using np.add.at for unbuffered addition
    np.add.at(grad_pts, nn_indices, weights[:, np.newaxis] * grid)
    
    # The gradient in R^3 is grad_pts.
    # However, the points are constrained to the sphere.
    # The true gradient on the manifold is the projection of grad_pts onto the tangent plane.
    # grad_tangent = grad_pts - (grad_pts . p) * p
    # But the optimizer moves in R^3. If we provide the R^3 gradient, the optimizer
    # will try to move in that direction. If we project it, the optimizer might drift off the sphere.
    # To keep points on the sphere, we can add a penalty term to the objective:
    # Penalty = sum (||p_i||^2 - 1)^2
    # Grad Penalty = 4 * (||p_i||^2 - 1) * p_i
    
    # Let's add a strong penalty to keep points on the sphere.
    norms_sq = np.sum(pts * pts, axis=1, keepdims=True)
    penalty = np.sum((norms_sq - 1.0) ** 2)
    obj += 100.0 * penalty
    
    grad_penalty = 4.0 * 100.0 * (norms_sq - 1.0) * pts
    
    grad_total = grad_pts + grad_penalty
    
    return obj, grad_total.flatten()

def _polish_with_hull(pts, deadline):
    """
    Final polish using the exact hull-based metric and subgradient-like steps.
    This ensures we don't get stuck in a local minimum of the surrogate that is
    worse than a nearby point in the true metric.
    """
    pts = pts.copy()
    best_r = _hull_radius(pts)
    best_pts = pts.copy()
    
    step = 0.01
    no_improve = 0
    
    while time.time() < deadline - 2.0 and no_improve < 50:
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            M = len(faces)
            
            A = pts[faces[:, 0]]
            B = pts[faces[:, 1]]
            C = pts[faces[:, 2]]
            
            AB = B - A
            AC = C - A
            cross = np.cross(AB, AC)
            area = 0.5 * np.linalg.norm(cross, axis=1)
            
            valid = area > _EPS
            if not np.any(valid):
                break
            
            normals = cross[valid].copy()
            norms = np.linalg.norm(normals, axis=1, keepdims=True)
            centroids = (A[valid] + B[valid] + C[valid]) / 3.0
            signs = np.sum(normals * centroids, axis=1, keepdims=True)
            normals = normals * np.sign(signs)
            normals = normals / (norms + _EPS)
            
            cos_theta = np.sum(normals * A[valid], axis=1)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            radii = np.arccos(cos_theta)
            
            curr_r = radii.max()
            if curr_r < best_r - 1e-10:
                best_r = curr_r
                best_pts = pts.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 10:
                    step *= 0.8
                    no_improve = 0
                    if step < 1e-5:
                        break
            
            worst_idx = np.argmax(radii)
            worst_face = faces[valid][worst_idx]
            pole = normals[worst_idx]
            
            cand = pts.copy()
            for v in worst_face:
                tangent_dir = pole - pts[v]
                tangent_dir = tangent_dir - np.dot(tangent_dir, pts[v]) * pts[v]
                norm_t = np.linalg.norm(tangent_dir)
                if norm_t > _EPS:
                    tangent_dir /= norm_t
                else:
                    continue
                cand[v] = pts[v] + step * tangent_dir
                cand[v] /= (np.linalg.norm(cand[v]) + _EPS)
            
            new_r = _hull_radius(cand)
            if new_r < curr_r:
                pts = cand
            else:
                step *= 0.9
                
        except Exception:
            break
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # 1. Generate dense grid for surrogate
    grid = _generate_grid(n=5000)
    
    # 2. Initial point: Fibonacci spiral
    pts_init = _initial_fib()
    flat_init = pts_init.flatten()
    
    # 3. Optimize using L-BFGS-B on the smooth surrogate
    best_pts = None
    best_r = np.pi
    
    # Run optimization
    try:
        res = minimize(
            _surrogate_objective_and_grad,
            flat_init,
            method='L-BFGS-B',
            jac=True, # We return (obj, grad)
            options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 20}
        )
        
        if res.success or res.fun < 100.0: # Check if it didn't explode
            pts_opt = res.x.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            pts_opt = pts_opt / (norms + _EPS)
            
            r_opt = _hull_radius(pts_opt)
            if r_opt < best_r:
                best_r = r_opt
                best_pts = pts_opt.copy()
                
    except Exception:
        pass
    
    # 4. If optimization failed or didn't improve, use initial
    if best_pts is None:
        best_pts = pts_init.copy()
        best_r = _hull_radius(best_pts)
        
    # 5. Polish with exact hull metric
    if time.time() < deadline - 5.0:
        best_pts, best_r = _polish_with_hull(best_pts, deadline)
        
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / (norms + _EPS)
    
    # Validity check
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
