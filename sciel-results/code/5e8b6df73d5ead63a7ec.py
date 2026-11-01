"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _calc_radius(pts):
    """Compute the maximum valid radius for given point positions."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_hex_lattice(n, spacing):
    """Generate a hexagonal lattice with given spacing, returning points in [-0.5, 0.5]^2."""
    row_h = spacing * np.sqrt(3.0) / 2.0
    col_w = spacing
    
    y_vals = np.arange(-0.5, 0.5 + row_h, row_h)
    all_pts = []
    
    for j, y in enumerate(y_vals):
        if y < -0.5 or y > 0.5:
            continue
        offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
        x_start = -0.5 + offset
        x_vals = np.arange(x_start, 0.5 + col_w, col_w)
        for x in x_vals:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                all_pts.append([x, y])
    
    all_pts = np.array(all_pts)
    if len(all_pts) < n:
        return None
    
    # Select n points: prefer central points
    dist_to_bnd = np.minimum(
        0.5 - np.abs(all_pts[:, 0]),
        0.5 - np.abs(all_pts[:, 1])
    )
    idx = np.argsort(dist_to_bnd)[::-1]
    return all_pts[idx[:n]]

def _generate_initial_points(n, n_spacings=20):
    """Generate initial points by trying multiple hexagonal spacings and selecting the best."""
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings around the expected optimal
    spacings = np.linspace(0.065, 0.082, n_spacings)
    
    for d in spacings:
        pts = _generate_hex_lattice(n, d)
        if pts is None or len(pts) < n:
            continue
        
        score = _calc_radius(pts)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    if best_points is None:
        # Fallback: regular grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    return best_points

def _smooth_min_distance(pts, r, beta=50.0):
    """
    Compute a smooth approximation of the minimum constraint violation.
    Uses log-sum-exp for pairwise distances and boundary distances.
    Returns (value, gradient) where value is the smooth approximation
    of min(0, min_pair_dist - 2r, min_boundary_dist - r).
    """
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    bdist = np.minimum(bx, by)
    
    # Smooth min of (dists - 2r) for all pairs
    # We want to maximize the minimum, so we minimize the max violation
    # violation_pair = 2r - dist (positive when violated)
    viol_pair = 2.0 * r - dists
    viol_pair = np.where(np.isinf(viol_pair), -np.inf, viol_pair)
    
    # Smooth max of violations using log-sum-exp
    max_viol_pair = np.max(viol_pair)
    if max_viol_pair > 0:
        # Use log-sum-exp for smooth max
        scaled = viol_pair / beta
        scaled = np.where(np.isinf(scaled), -1e10, scaled)
        log_sum = np.log(np.sum(np.exp(scaled - np.max(scaled)))) + np.max(scaled)
        smooth_max_pair = beta * log_sum
        # Gradient of smooth max wrt each dist
        weights_pair = np.exp(scaled - np.max(scaled)) / np.sum(np.exp(scaled - np.max(scaled)))
        weights_pair = np.where(np.isinf(weights_pair), 0.0, weights_pair)
    else:
        smooth_max_pair = 0.0
        weights_pair = np.zeros_like(viol_pair)
    
    # Boundary violations
    viol_bound = r - bdist
    max_viol_bound = np.max(viol_bound)
    if max_viol_bound > 0:
        scaled_b = viol_bound / beta
        log_sum_b = np.log(np.sum(np.exp(scaled_b - np.max(scaled_b)))) + np.max(scaled_b)
        smooth_max_bound = beta * log_sum_b
        weights_bound = np.exp(scaled_b - np.max(scaled_b)) / np.sum(np.exp(scaled_b - np.max(scaled_b)))
    else:
        smooth_max_bound = 0.0
        weights_bound = np.zeros_like(viol_bound)
    
    # Total smooth violation
    total_viol = max(smooth_max_pair, smooth_max_bound)
    
    # Compute gradient
    grad = np.zeros_like(pts)
    
    # Pairwise gradient contribution
    # d(viol_pair)/d(pts_i) = -diff_ij / dist_ij for each j
    # d(smooth_max)/d(viol_pair_ij) = weights_pair_ij
    # So d(smooth_max)/d(pts_i) = -sum_j weights_pair_ij * diff_ij / dist_ij
    safe_dists = np.where(dists > 1e-10, dists, 1e-10)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # grad_i = -sum_j w_ij * (p_j - p_i) / dist_ij * (-1) = sum_j w_ij * (p_j - p_i) / dist_ij
    # Wait: viol = 2r - dist, d(viol)/d(p_i) = -d(dist)/d(p_i) = -(p_i - p_j)/dist = (p_j - p_i)/dist
    # d(smooth_max)/d(p_i) = sum_j w_ij * (p_j - p_i)/dist_ij
    pair_grad = np.zeros_like(pts)
    for i in range(n):
        for j in range(n):
            if i != j and weights_pair[i, j] > 1e-10:
                pair_grad[i] += weights_pair[i, j] * (pts[j] - pts[i]) / safe_dists[i, j]
    
    # Vectorized version
    weighted_diff = weights_pair[:, :, np.newaxis] * unit_vecs
    pair_grad = np.sum(weighted_diff, axis=1)
    
    # Boundary gradient
    # d(viol_bound)/d(p_i) = -d(bdist)/d(p_i)
    # If bx < by: bdist = bx = 0.5 - |x_i|, d(bx)/dx_i = -sign(x_i)
    # If by < bx: bdist = by = 0.5 - |y_i|, d(by)/dy_i = -sign(y_i)
    bound_grad = np.zeros_like(pts)
    active_x = bx <= by
    for i in range(n):
        if weights_bound[i] > 1e-10:
            if active_x[i]:
                bound_grad[i, 0] = weights_bound[i] * np.sign(pts[i, 0])
            else:
                bound_grad[i, 1] = weights_bound[i] * np.sign(pts[i, 1])
    
    total_grad = pair_grad + bound_grad
    
    return total_viol, total_grad

def _qp_optimize(points, r_init=None, max_iter=10000, lr_init=0.005, lr_decay=0.9997, 
                 lr_min=0.00001, beta=50.0, r_step=0.00005):
    """
    Optimize point positions and radius to maximize the minimum valid radius.
    Uses projected gradient descent on the smooth violation function.
    """
    n = len(points)
    if r_init is None:
        r = _calc_radius(points)
    else:
        r = r_init
    points = points.copy()
    
    best_points = points.copy()
    best_r = r
    
    for it in range(max_iter):
        lr = max(lr_init * (lr_decay ** it), lr_min)
        
        # Compute smooth violation and gradient
        viol, grad = _smooth_min_distance(points, r, beta=beta)
        
        # Update positions
        points = points - lr * grad
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Compute actual radius
        current_r = _calc_radius(points)
        
        # Track best
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        # Adaptive radius update
        if current_r >= r:
            r = current_r + r_step * 0.5
            r = min(r, current_r + 1e-7)
        else:
            r = max(current_r, r - r_step * 0.3)
        
        # Early stopping if no improvement
        if it % 100 == 0:
            pass  # Could add early stopping logic
    
    return best_points

def _qp_optimize_v2(points, r_init=None, max_iter=15000, lr_init=0.003, lr_decay=0.9998,
                    lr_min=0.000005, beta=80.0, r_step=0.00003):
    """
    Improved QP optimizer with:
    - Smaller initial learning rate for stability
    - Slower decay for more iterations at useful learning rates
    - Smaller radius increment for finer convergence
    - Higher beta for smoother approximation
    """
    n = len(points)
    if r_init is None:
        r = _calc_radius(points)
    else:
        r = r_init
    points = points.copy()
    
    best_points = points.copy()
    best_r = r
    no_improve_count = 0
    patience = 2000
    
    for it in range(max_iter):
        lr = max(lr_init * (lr_decay ** it), lr_min)
        
        # Compute smooth violation and gradient
        viol, grad = _smooth_min_distance(points, r, beta=beta)
        
        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        # Update positions
        points = points - lr * grad
        
        # Project back into square with small margin
        points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-9, 0.5 - 1e-9)
        points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Compute actual radius
        current_r = _calc_radius(points)
        
        # Track best
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= patience:
            break
        
        # Adaptive radius update
        if current_r >= r:
            r = current_r + r_step * 0.4
            r = min(r, current_r + 1e-8)
        else:
            r = max(current_r, r - r_step * 0.2)
    
    return best_points

def _polish(points, max_iter=5000, lr_init=0.0005, lr_decay=0.9999, lr_min=0.000001, beta=100.0):
    """Final polishing phase with very small steps for high precision."""
    n = len(points)
    r = _calc_radius(points)
    points = points.copy()
    
    best_points = points.copy()
    best_r = r
    
    for it in range(max_iter):
        lr = max(lr_init * (lr_decay ** it), lr_min)
        
        viol, grad = _smooth_min_distance(points, r, beta=beta)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm
        
        points = points - lr * grad
        points[:, 0] = np.clip(points[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        points[:, 1] = np.clip(points[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        current_r = _calc_radius(points)
        if current_r > best_r:
            best_r = current_r
            best_points = points.copy()
        
        if current_r >= r:
            r = min(current_r + 1e-9, current_r + 1e-9)
        else:
            r = max(current_r, r - 1e-10)
    
    return best_points

def construct():
    # Step 1: Generate initial hexagonal lattice
    points = _generate_initial_points(N)
    
    # Step 2: Main optimization
    points = _qp_optimize_v2(points, max_iter=15000, lr_init=0.003, lr_decay=0.9998,
                            lr_min=0.000005, beta=80.0, r_step=0.00003)
    
    # Step 3: Final polish with very small steps
    points = _polish(points, max_iter=5000, lr_init=0.0005, lr_decay=0.9999,
                     lr_min=0.000001, beta=100.0)
    
    return points
# EVOLVE-BLOCK-END
