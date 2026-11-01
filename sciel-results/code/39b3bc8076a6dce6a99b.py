"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import random

def _calc_radius(pts):
    """Compute the maximum valid radius for given point positions."""
    if len(pts) == 0:
        return 0.0
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    boundary_dists = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair / 2.0, min_boundary)

def _generate_hex_lattice_points(n):
    """
    Generate points from a hexagonal lattice and select n points using 
    Farthest Point Sampling to ensure uniform coverage.
    """
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings
    spacings = np.linspace(0.065, 0.085, 20)
    
    for d in spacings:
        row_h = d * np.sqrt(3.0) / 2.0
        col_w = d
        
        # Generate all lattice points in a slightly larger box to allow for shifts
        # We will try a few shifts to align better with boundaries
        shifts = [(0,0), (d/2, 0), (0, row_h/2), (d/2, row_h/2)]
        
        for sx, sy in shifts:
            y_start = -0.5 - row_h + sy
            y_vals = np.arange(y_start, 0.5 + row_h + row_h, row_h)
            all_pts = []
            
            for j, y in enumerate(y_vals):
                if y < -0.5 - d or y > 0.5 + d:
                    continue
                # Offset for hex packing
                offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
                x_start = -0.5 - col_w + offset + sx
                x_vals = np.arange(x_start, 0.5 + col_w + offset + col_w, col_w)
                for x in x_vals:
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        all_pts.append([x, y])
            
            if len(all_pts) < n:
                continue
            
            all_pts = np.array(all_pts)
            
            # Farthest Point Sampling to select n points
            # Start with a random point or center
            selected = []
            # Use center as first point for stability
            center_idx = np.argmin(np.sum((all_pts - [0,0])**2, axis=1))
            selected.append(all_pts[center_idx])
            
            # Keep a set of selected points for distance calculation
            # For efficiency, we can just maintain a list of distances
            min_dists = np.full(len(all_pts), np.inf)
            min_dists[center_idx] = 0
            
            for _ in range(n - 1):
                # Pick the point with the maximum minimum distance to already selected
                idx = np.argmax(min_dists)
                if min_dists[idx] == 0:
                    break
                selected.append(all_pts[idx])
                
                # Update min_dists for all other points
                diff = all_pts - all_pts[idx]
                dists_to_new = np.sqrt(np.sum(diff**2, axis=1))
                min_dists = np.minimum(min_dists, dists_to_new)
                min_dists[idx] = 0 # Exclude selected point
                
            if len(selected) < n:
                continue
                
            selected = np.array(selected)
            score = _calc_radius(selected)
            
            if score > best_score:
                best_score = score
                best_points = selected.copy()
    
    if best_points is None:
        # Fallback: regular grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n]
    
    return best_points

def _qp_optimize(points, max_iter=2000, lr_init=0.005, momentum=0.9):
    """
    Optimize point positions using Projected Gradient Descent on a 
    smoothed energy function (sum of squared violations).
    """
    n = len(points)
    points = points.copy()
    velocity = np.zeros_like(points)
    
    # Initial radius estimate
    r = _calc_radius(points)
    # Start with a slightly smaller r to allow movement
    r_target = r * 0.95
    
    for it in range(max_iter):
        # Decay learning rate
        lr = lr_init * (0.995 ** it)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        b_dist = np.minimum(bx, by)
        
        # --- Calculate Gradients ---
        grad = np.zeros_like(points)
        
        # 1. Pairwise Repulsion Gradient
        # We want to minimize sum_{i<j} max(0, 2*r_target - d_ij)^2
        # Let v_ij = 2*r_target - d_ij
        # grad_i = sum_j 2 * v_ij * (1/d_ij) * (p_i - p_j)  (if v_ij > 0)
        
        # Vectorized calculation
        # v = 2*r_target - dists
        v = 2.0 * r_target - dists
        # Mask for violations
        mask = v > 0
        # To avoid division by zero, use a small epsilon
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        
        # Gradient contribution from pair (i,j) to i is:
        # 2 * v_ij * (p_i - p_j) / d_ij
        # Note: d_ij = |p_i - p_j|
        # (p_i - p_j) / d_ij is the unit vector from j to i.
        # So we push i away from j.
        
        # We need to sum over all j for each i.
        # term_ij = 2 * v_ij * (p_i - p_j) / d_ij
        # This is tricky to vectorize directly as a matrix product because of the denominator.
        
        # Alternative:
        # grad_i = sum_j w_ij * (p_i - p_j)
        # where w_ij = 2 * v_ij / d_ij if v_ij > 0 else 0
        
        weights = np.zeros_like(dists)
        weights[mask] = (2.0 * v[mask]) / safe_dists[mask]
        
        # grad_i = sum_j weights[i,j] * (p_i - p_j)
        #        = p_i * sum_j weights[i,j] - sum_j weights[i,j] * p_j
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        pair_grad = row_sums[:, np.newaxis] * points - weighted_sum
        
        # 2. Boundary Repulsion Gradient
        # Minimize sum_i max(0, r_target - b_i)^2
        # b_i = min(bx_i, by_i)
        # If bx_i < by_i, b_i = bx_i = 0.5 - |x_i|
        # grad_x of (r - (0.5 - |x|))^2 = 2(r - b) * sign(x) (if active)
        
        active_x = bx <= by
        active_y = ~active_x
        
        viol_x = np.where(active_x, np.maximum(0.0, r_target - bx), 0.0)
        viol_y = np.where(active_y, np.maximum(0.0, r_target - by), 0.0)
        
        grad_x = 2.0 * viol_x * np.sign(points[:, 0])
        grad_y = 2.0 * viol_y * np.sign(points[:, 1])
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        # --- Update with Momentum ---
        velocity = momentum * velocity - lr * total_grad
        points = points + velocity
        
        # Project back into square
        points[:, 0] = np.clip(points[:, 0], -0.5, 0.5)
        points[:, 1] = np.clip(points[:, 1], -0.5, 0.5)
        
        # --- Update Target Radius ---
        current_r = _calc_radius(points)
        
        # If we improved, increase target r slightly
        if current_r > r_target:
            r_target = current_r + 0.0001
        else:
            # If we lost radius, decrease target r to recover
            # But don't drop it too fast
            r_target = max(current_r, r_target - 0.00005)
            
        # Optional: Stop if no improvement for a while? 
        # The loop runs to max_iter, which is fine.
        
    return points

def construct():
    # Step 1: Generate initial points using Hex Lattice + FPS
    points = _generate_hex_lattice_points(N)
    
    # Step 2: Optimize using QP-style projected gradient descent
    # Run two stages: coarse then fine
    points = _qp_optimize(points, max_iter=3000, lr_init=0.008, momentum=0.9)
    points = _qp_optimize(points, max_iter=2000, lr_init=0.002, momentum=0.9)
    
    return points
# EVOLVE-BLOCK-END
