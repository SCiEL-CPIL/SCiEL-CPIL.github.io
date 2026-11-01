"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np

def _calc_radius(pts):
    """Compute the maximum valid radius for given point positions."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_hex_lattice(n, aspect_ratio=1.0):
    """
    Generate hexagonal lattice points within the unit square.
    aspect_ratio controls the stretching in x-direction.
    """
    # Determine spacing to fit approximately n points
    # Hexagonal packing density: pi / (2 * sqrt(3)) ≈ 0.9069
    # Area per point in hex lattice: d^2 * sqrt(3) / 2
    # For n points in unit square: n * d^2 * sqrt(3)/2 ≈ 1 => d ≈ sqrt(2/(n*sqrt(3)))
    d = np.sqrt(2.0 / (n * np.sqrt(3.0)))
    
    row_h = d * np.sqrt(3.0) / 2.0
    col_w = d * aspect_ratio
    
    # Generate lattice points
    y_vals = np.arange(-0.5, 0.5 + row_h, row_h)
    all_pts = []
    
    for j, y in enumerate(y_vals):
        if y < -0.5 - 0.01 or y > 0.5 + 0.01:
            continue
        offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
        x_start = -0.5 - col_w + offset
        x_vals = np.arange(x_start, 0.5 + col_w + offset + col_w, col_w)
        for x in x_vals:
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                all_pts.append([x, y])
    
    if len(all_pts) < n:
        return None
    
    all_pts = np.array(all_pts)
    
    # Select n points by keeping those closest to center
    dist_to_center = np.sqrt(all_pts[:, 0]**2 + all_pts[:, 1]**2)
    idx = np.argsort(dist_to_center)[:n]
    selected = all_pts[idx]
    
    return selected

def _smooth_min_gradient(pts, r, smooth=0.002):
    """
    Compute smooth gradient for the objective: maximize min(pairwise_dist/2, boundary_dist)
    Using log-sum-exp approximation for smoothness.
    """
    n = len(pts)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    dists = np.sqrt(np.maximum(dists_sq, 1e-12))
    np.fill_diagonal(dists, np.inf)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Boundary distances
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    boundary = np.minimum(bx, by)
    
    # Combine all constraints: pairwise distances (divided by 2) and boundary distances
    # We want to maximize the minimum of these
    # Use log-sum-exp for smooth approximation of min
    
    # Pairwise contribution: dists/2
    pair_vals = dists / 2.0
    pair_vals = np.where(np.isinf(pair_vals), 10.0, pair_vals)  # Replace inf with large value
    
    # All values: pairwise (n*(n-1)/2 values) + boundary (n values)
    # For efficiency, only consider near-violating pairs
    threshold = 2.0 * r
    near_mask = dists < threshold * 1.5
    
    # Weighted gradient: pairs closer to violation contribute more
    safe_dists = np.where(dists > 1e-10, dists, 1e-10)
    viol = np.maximum(0.0, threshold - dists)
    
    # Smooth weight: exp(-(dist - threshold)^2 / (2*smooth^2)) for pairs near threshold
    weight_pair = np.zeros_like(dists)
    mask = near_mask & (dists < np.inf)
    weight_pair[mask] = np.exp(-((dists[mask] - threshold) ** 2) / (2.0 * smooth ** 2))
    
    # Normalize weights
    total_w = np.sum(weight_pair, axis=1, keepdims=True)
    total_w = np.where(total_w > 1e-10, total_w, 1e-10)
    norm_weight = weight_pair / total_w
    
    # Pairwise gradient: grad_i = sum_j w_ij * (p_j - p_i) / dist_ij
    # Actually for maximizing min dist, gradient points away from other points
    # grad_i = sum_j (w_ij / dist_ij) * (p_i - p_j)
    
    inv_dists = np.where(dists > 1e-10, 1.0 / dists, 0.0)
    grad_weight = norm_weight * inv_dists
    
    row_sums = np.sum(grad_weight, axis=1)
    weighted_sum = grad_weight @ pts
    pair_grad = row_sums[:, np.newaxis] * pts - weighted_sum
    
    # Boundary gradient
    active_x = bx <= by
    viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
    viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
    
    # Smooth boundary weight
    bw_smooth = 0.003
    weight_bx = np.exp(-((r - bx) ** 2) / (2.0 * bw_smooth ** 2)) * active_x
    weight_by = np.exp(-((r - by) ** 2) / (2.0 * bw_smooth ** 2)) * (~active_x)
    
    total_bw = weight_bx + weight_by
    total_bw = np.where(total_bw > 1e-10, total_bw, 1e-10)
    norm_bw = (weight_bx + weight_by) / total_bw
    
    grad_x = norm_bw * np.sign(pts[:, 0])
    grad_y = norm_bw * np.sign(pts[:, 1])
    
    bound_grad = np.column_stack([grad_x, grad_y])
    
    return pair_grad + bound_grad

def _qp_optimize(points, max_iter=8000, lr_init=0.005, lr_min=0.00002, decay=0.9995):
    """
    Optimize point positions using projected gradient descent.
    Maximizes the minimum distance between circles.
    """
    n = len(points)
    points = points.copy()
    r = _calc_radius(points)
    best_r = r
    best_points = points.copy()
    
    # Adaptive smooth parameter
    smooth_init = 0.003
    smooth_min = 0.0005
    
    no_improve_count = 0
    patience = 500
    
    for it in range(max_iter):
        lr = max(lr_init * (decay ** it), lr_min)
        smooth = max(smooth_init * (0.999 ** it), smooth_min)
        
        # Compute gradient
        grad = _smooth_min_gradient(points, r, smooth)
        
        # Gradient clipping to prevent large steps
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        max_grad = 10.0
        grad = np.where(grad_norm > max_grad, grad * (max_grad / grad_norm), grad)
        
        # Update
        points = points - lr * grad
        
        # Project to square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # Evaluate
        current_r = _calc_radius(points)
        
        if current_r > best_r + 1e-9:
            best_r = current_r
            best_points = points.copy()
            no_improve_count = 0
            # Slightly increase target radius
            r = min(current_r + 0.00005, current_r + 0.0001)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                # Reset to best and reduce learning rate
                points = best_points.copy()
                lr_init *= 0.5
                decay = min(decay * 0.999, 0.99999)
                no_improve_count = 0
        
        # Early stopping if convergence
        if lr < lr_min * 1.1 and no_improve_count > 100:
            break
    
    return best_points

def construct():
    """
    Main function: generate hexagonal lattice and optimize.
    """
    # Try multiple aspect ratios and initial configurations
    best_points = None
    best_score = -1.0
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for ar in aspect_ratios:
        pts = _generate_hex_lattice(N, aspect_ratio=ar)
        if pts is None:
            continue
        
        # Optimize
        pts_opt = _qp_optimize(pts, max_iter=6000, lr_init=0.004, lr_min=0.00003, decay=0.9996)
        
        score = _calc_radius(pts_opt)
        if score > best_score:
            best_score = score
            best_points = pts_opt.copy()
    
    # Final polish with very small steps
    if best_points is not None:
        best_points = _qp_optimize(best_points, max_iter=3000, lr_init=0.001, lr_min=0.00001, decay=0.9999)
    
    # Fallback if all failed
    if best_points is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:N]
    
    return best_points
# EVOLVE-BLOCK-END
