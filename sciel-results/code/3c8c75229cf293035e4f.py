import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _compute_radius(pts):
    """Compute the valid radius for a set of points."""
    if pts.shape != (N, 2):
        return 0.0
    # Check finite
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    if np.any(pts < -0.5) or np.any(pts > 0.5):
        # If any point is outside, radius is effectively 0 or invalid
        # The problem statement says "inside the square", usually implying strict or non-strict.
        # "inside the square [-1/2, 1/2]^2"
        # If a point is outside, it's invalid.
        # However, during optimization, we might project.
        # For scoring, if a point is outside, the distance to boundary is negative.
        # The formula: min(1/2 - |c_ik|). If c_ik > 0.5, this is negative.
        # Radius must be positive. So invalid.
        # We will ensure our optimizer keeps points inside.
        pass
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    
    min_boundary_dist = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top], axis=1), axis=1))
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def _soft_min_objective(x, beta):
    """
    Objective to minimize: -soft_min(distances).
    We want to maximize the minimum distance.
    soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
    So we minimize: 1/beta * log(sum(exp(-beta * d)))
    
    The distances are:
    1. Pairwise distances d_ij
    2. Distances to boundaries b_i
    
    To handle boundaries correctly in the sum:
    We add terms for each boundary distance.
    """
    pts = x.reshape(N, 2)
    
    # Pairwise
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    # Boundary
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    
    # For numerical stability in log-sum-exp
    # We want to compute log(sum(exp(-beta * d)))
    # Let M = min(d). Then sum(exp(-beta * d)) = exp(-beta * M) * sum(exp(-beta * (d - M)))
    # log(sum(...)) = -beta * M + log(sum(exp(-beta * (d - M))))
    
    # Collect all distances into a flat array for easy min calculation
    # We only need the min for the shift, but we need all for the sum.
    # To save memory/time, we can compute in chunks or just use the arrays.
    
    # Flatten pairwise (excluding diagonal)
    # Upper triangle indices
    iu = np.triu_indices(N, k=1)
    d_pairs = dist[iu]
    
    d_bounds = np.concatenate([d_left, d_right, d_bottom, d_top])
    
    # Combine all distances
    # Note: Boundary distances are 'direct' distances to wall.
    # Pair distances are 'direct' distances between centers.
    # The radius r is min(d_pair/2, d_bound).
    # So the effective distance for pair is 2*r? No.
    # If we define the objective as maximizing r,
    # constraint d_pair >= 2r and d_bound >= r.
    # Let's normalize all distances to be comparable to r.
    # Let D_pairs = d_pairs / 2.0
    # Let D_bounds = d_bounds
    # We want to maximize min(D_pairs, D_bounds).
    
    D_pairs = d_pairs / 2.0
    D_bounds = d_bounds
    
    all_D = np.concatenate([D_pairs, D_bounds])
    
    # Soft min
    # obj = -soft_min(all_D) = 1/beta * log(sum(exp(-beta * all_D)))
    
    # Numerical stability
    min_D = np.min(all_D)
    if min_D < 0:
        # Points are overlapping or outside. Penalty should be high.
        # exp(-beta * negative) is large.
        pass
        
    # To avoid overflow in exp(-beta * (D - min_D)) when beta is large and D-min_D is large positive?
    # D - min_D >= 0. exp(-beta * positive) <= 1. So no overflow.
    # Underflow is fine (becomes 0).
    
    shifted = -beta * (all_D - min_D)
    # Clip shifted to avoid underflow issues if needed, but exp handles it.
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    
    log_sum = np.log(sum_exp)
    
    obj = (1.0 / beta) * (-beta * min_D + log_sum)
    # obj = -min_D + (1/beta) * log_sum_exp
    # As beta -> inf, obj -> -min_D. Minimizing obj -> Maximizing min_D.
    
    return obj

def _soft_min_gradient(x, beta):
    """
    Gradient of the objective function.
    """
    pts = x.reshape(N, 2)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    
    d_left = pts[:, 0] + 0.5
    d_right = 0.5 - pts[:, 0]
    d_bottom = pts[:, 1] + 0.5
    d_top = 0.5 - pts[:, 1]
    
    D_pairs = dist / 2.0
    np.fill_diagonal(D_pairs, np.inf)
    
    D_bounds = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    
    # We need the weights w_k = exp(-beta * D_k) / sum(exp(-beta * D_k))
    # where D_k are the normalized distances.
    
    # For efficiency, we can compute the gradient directly.
    # grad_x_i = sum_k w_k * (d D_k / d x_i)
    
    # D_pairs_ij = dist_ij / 2
    # d D_pairs_ij / d x_i = - (x_i - x_j) / (2 * dist_ij)
    
    # D_bounds is linear.
    
    # Let's compute the unnormalized weights first.
    # To save memory, we don't want to create a huge (N*N + 4N) array of exps if possible, 
    # but N=200 is small.
    
    # Pairwise part
    # dist is (N, N)
    # D_pairs is (N, N) with inf on diagonal.
    
    # We need sum over all pairs.
    # Let's compute the contribution to gradient.
    
    # grad_pair = -0.5 * sum_j w_ij * (x_i - x_j) / dist_ij
    # where w_ij = exp(-beta * dist_ij / 2)
    
    # Boundary part
    # grad_bound_i = w_left * (1, 0) + w_right * (-1, 0) + w_bottom * (0, 1) + w_top * (0, -1)
    
    # Total weight normalization Z = sum_pairs(exp(-beta * dist_ij / 2)) + sum_bounds(exp(-beta * d_bound))
    
    # Compute exp terms
    # For pairs:
    exp_pairs = np.exp(-beta * D_pairs)
    np.fill_diagonal(exp_pairs, 0.0)
    
    exp_bounds = np.exp(-beta * D_bounds)
    
    Z = np.sum(exp_pairs) + np.sum(exp_bounds)
    
    if Z < 1e-10:
        return np.zeros_like(x)
    
    # Normalize weights
    w_pairs = exp_pairs / Z
    w_bounds = exp_bounds / Z
    
    # Gradient from pairs
    # grad_i = -0.5 * sum_j w_ij * (x_i - x_j) / dist_ij
    # u_ij = (x_i - x_j) / dist_ij
    # We have diff = x_i - x_j.
    # u = diff / dist[:, :, np.newaxis]
    # Handle dist=0 (should be inf on diagonal, but off-diagonal could be 0 if coincident)
    dist_safe = np.where(dist > 1e-10, dist, 1e-10)
    u = diff / dist_safe[:, :, np.newaxis]
    
    # sum_j w_ij * u_ij
    # w_pairs is (N, N)
    # u is (N, N, 2)
    grad_pairs = -0.5 * np.sum(w_pairs[:, :, np.newaxis] * u, axis=1)
    
    # Gradient from bounds
    # w_bounds is (N, 4)
    # cols: left, right, bottom, top
    # d_left/dx = 1, d_right/dx = -1
    # d_bottom/dy = 1, d_top/dy = -1
    grad_bounds_x = w_bounds[:, 0] - w_bounds[:, 1]
    grad_bounds_y = w_bounds[:, 2] - w_bounds[:, 3]
    
    grad_bounds = np.column_stack((grad_bounds_x, grad_bounds_y))
    
    grad = grad_pairs + grad_bounds
    
    return grad.reshape(-1)

def _initialize_hexagonal(rows, cols):
    """Generate a hexagonal grid."""
    # Calculate spacing
    # Fit into [-0.5, 0.5]
    # Height = (rows - 1) * dy
    # Width = (cols - 1) * dx + dx/2 (stagger)
    
    # Try to maximize spacing
    # dy = dx * sqrt(3) / 2
    # (rows - 1) * dx * sqrt(3) / 2 <= 1.0  => dx <= 2 / ((rows - 1) * sqrt(3))
    # (cols - 0.5) * dx <= 1.0 => dx <= 1 / (cols - 0.5)
    
    dx_h = 2.0 / ((rows - 1) * np.sqrt(3)) if rows > 1 else 1.0
    dx_w = 1.0 / (cols - 0.5)
    dx = min(dx_h, dx_w)
    dy = dx * np.sqrt(3) / 2.0
    
    points = []
    for i in range(rows):
        y = (i - (rows - 1) / 2.0) * dy
        for j in range(cols):
            x = (j - (cols - 1) / 2.0) * dx
            if i % 2 != 0:
                x += dx / 2.0
            points.append([x, y])
            
    pts = np.array(points)
    
    # Trim or pad to N
    if len(pts) > N:
        # Remove points from the edges? Or just take first N?
        # Taking first N might be biased. 
        # Let's take the N points closest to the center of the grid?
        # No, we want a rectangular block.
        # Just take the first N in raster order.
        pts = pts[:N]
    elif len(pts) < N:
        # Add random points?
        # This shouldn't happen if we pick rows/cols correctly.
        # For N=200, 14x15=210. 13x16=208.
        # We'll handle by padding with small random perturbations of existing points?
        # Or just resize the grid?
        # Let's assume we pick dimensions that yield >= N.
        pass
        
    # Center
    pts -= pts.mean(axis=0)
    
    # Scale to fit
    max_extent = np.max(np.abs(pts))
    if max_extent > 0.49:
        pts *= (0.49 / max_extent)
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    best_points = None
    best_radius = 0.0
    
    # Try a few different hexagonal grid configurations
    # N=200
    # 14 rows, 15 cols -> 210
    # 13 rows, 16 cols -> 208
    # 12 rows, 17 cols -> 204
    # 11 rows, 19 cols -> 209
    
    configs = [
        (14, 15),
        (13, 16),
        (12, 17),
        (15, 14) # Wider than tall?
    ]
    
    # Also try a simple square grid?
    # 15x14 = 210
    
    for rows, cols in configs:
        if time.time() > deadline:
            break
            
        init_pts = _initialize_hexagonal(rows, cols)
        
        # Add small random noise to break symmetry
        noise = np.random.normal(0, 0.001, init_pts.shape)
        init_pts += noise
        # Clip to square
        init_pts = np.clip(init_pts, -0.49, 0.49)
        
        x0 = init_pts.flatten()
        
        # Optimize with L-BFGS-B
        # We will run a few phases with increasing beta
        
        current_pts = init_pts.copy()
        
        # Phase 1: Low beta for global movement
        # Phase 2: High beta for local refinement
        
        phases = [
            (10.0, 100),   # beta, max_iter
            (50.0, 100),
            (200.0, 100),
            (1000.0, 100)
        ]
        
        for beta, max_iter in phases:
            if time.time() > deadline:
                break
                
            # Define objective and gradient for this beta
            def obj(x):
                return _soft_min_objective(x, beta)
                
            def grad(x):
                return _soft_min_gradient(x, beta)
                
            # Bounds: each coordinate in [-0.5, 0.5]
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            try:
                res = minimize(
                    obj, 
                    current_pts.flatten(), 
                    jac=grad, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                current_pts = res.x.reshape(N, 2)
            except Exception:
                # If optimization fails, keep current points
                pass
                
        # Evaluate
        r = _compute_radius(current_pts)
        if r > best_radius:
            best_radius = r
            best_points = current_pts.copy()
            
    if best_points is None:
        # Fallback
        best_points = _initialize_hexagonal(14, 15)
        
    return best_points
