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

def _generate_dense_hex(n):
    """
    Generate a dense hexagonal lattice and select n points optimally.
    Tries multiple spacings and offsets to find a better initial configuration.
    """
    best_points = None
    best_score = -1.0
    
    # Try a range of spacings. Optimal d is ~ 1/sqrt(200) * sqrt(2/sqrt(3)) ? 
    # Area per point ~ 0.005. Hex area ~ d^2 * sqrt(3)/2. d ~ sqrt(0.005 * 2 / 1.732) ~ 0.075.
    spacings = np.linspace(0.072, 0.078, 10)
    
    # Try different vertical offsets to shift the grid
    offsets = [0.0, 0.1, 0.2, 0.3, 0.4]
    
    for d in spacings:
        row_h = d * np.sqrt(3.0) / 2.0
        col_w = d
        
        for v_offset in offsets:
            # Generate all lattice points in [-0.5-d, 0.5+d]^2
            y_vals = np.arange(-0.5 - row_h + v_offset, 0.5 + row_h + row_h, row_h)
            all_pts = []
            
            for j, y in enumerate(y_vals):
                if y < -0.5 - d or y > 0.5 + d:
                    continue
                offset = (col_w / 2.0) if (j % 2 != 0) else 0.0
                x_start = -0.5 - col_w + offset
                x_vals = np.arange(x_start, 0.5 + col_w + offset + col_w, col_w)
                for x in x_vals:
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        all_pts.append([x, y])
            
            if len(all_pts) < n:
                continue
            
            all_pts = np.array(all_pts)
            
            # Selection Strategy:
            # 1. Calculate distance to boundary for each point.
            # 2. We want to keep points that are far from the boundary.
            # 3. However, simply taking the top N central points might leave "holes" 
            #    that prevent a larger radius from being achieved due to local constraints.
            #    A better heuristic: sort by distance to boundary, take top N.
            
            dist_to_bnd = np.minimum(
                0.5 - np.abs(all_pts[:, 0]),
                0.5 - np.abs(all_pts[:, 1])
            )
            
            idx = np.argsort(dist_to_bnd)[::-1]
            selected = all_pts[idx[:n]]
            
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

def _qp_optimize(points, max_iter=5000):
    """
    Optimize point positions to maximize minimum distance using
    projected gradient descent with adaptive radius increase and
    simulated annealing-style perturbations to escape local minima.
    """
    n = len(points)
    r = _calc_radius(points)
    best_r = r
    best_points = points.copy()
    
    # Parameters
    lr_init = 0.005
    lr_min = 0.00001
    beta = 0.998  # Decay factor
    r_step = 0.0002  # How much to try to increase r each successful iteration
    patience = 100
    no_improve = 0
    
    points = points.copy()
    
    for it in range(max_iter):
        lr = max(lr_init * (beta ** it), lr_min)
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(np.maximum(dists_sq, 1e-12))
        np.fill_diagonal(dists, np.inf)
        np.fill_diagonal(dists_sq, np.inf)
        
        # --- Pairwise gradient contribution ---
        # We want to maximize the minimum distance.
        # Approximate objective: minimize sum of (2r - d_ij)^2 for d_ij < 2r
        # Gradient of (2r - d)^2 w.r.t p_i is -2(2r-d) * (p_i - p_j)/d
        # So we add 2(2r-d)/d * (p_j - p_i) to p_i
        
        mask = dists < 2.0 * r
        safe_dists = np.where(dists > 1e-10, dists, 1e-10)
        
        # Violation amount
        viol = np.where(mask, 2.0 * r - dists, 0.0)
        
        # Weight for each pair: proportional to violation
        # Using 2 * viol / dist
        weights = np.zeros_like(dists)
        weights[mask] = (2.0 * viol[mask]) / safe_dists[mask]
        
        # Gradient from pairwise constraints
        # grad_i = sum_j w_ij * (p_j - p_i)
        row_sums = np.sum(weights, axis=1)
        weighted_sum = weights @ points
        pair_grad = weighted_sum - row_sums[:, np.newaxis] * points
        
        # --- Boundary gradient contribution ---
        # Objective: minimize sum of (r - b_i)^2 for b_i < r
        # b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
        # If active constraint is x: b_i = 0.5 - |x_i|. grad_b/grad_x = -sign(x_i).
        # grad_obj/grad_x = -2(r-b_i) * (-sign(x_i)) = 2(r-b_i)sign(x_i)
        # We want to move in direction of -grad_obj, so -2(r-b_i)sign(x_i)
        
        bx = 0.5 - np.abs(points[:, 0])
        by = 0.5 - np.abs(points[:, 1])
        
        # Active boundary constraint per point
        active_x = bx <= by
        
        viol_x = np.where(active_x, np.maximum(0.0, r - bx), 0.0)
        viol_y = np.where(~active_x, np.maximum(0.0, r - by), 0.0)
        
        # Gradient direction to move (negative of gradient of penalty)
        grad_x = -2.0 * viol_x * np.sign(points[:, 0])
        grad_y = -2.0 * viol_y * np.sign(points[:, 1])
        
        bound_grad = np.column_stack([grad_x, grad_y])
        
        total_grad = pair_grad + bound_grad
        
        # Update positions
        points_new = points - lr * total_grad
        
        # Project back into square
        points_new[:, 0] = np.clip(points_new[:, 0], -0.5, 0.5)
        points_new[:, 1] = np.clip(points_new[:, 1], -0.5, 0.5)
        
        # Check current radius
        current_r = _calc_radius(points_new)
        
        # Adaptive radius update
        if current_r >= best_r:
            # We improved or matched. Update best.
            best_r = current_r
            best_points = points_new.copy()
            no_improve = 0
            # Increase r target slightly to encourage further expansion
            r = min(r + r_step, current_r + 1e-5)
        else:
            # We did not improve.
            no_improve += 1
            # If stuck, perturb
            if no_improve > patience:
                # Add small random noise
                noise_scale = 0.001 * (1.0 + no_improve / patience)
                points_new = points_new + np.random.normal(0, noise_scale, points_new.shape)
                points_new[:, 0] = np.clip(points_new[:, 0], -0.5, 0.5)
                points_new[:, 1] = np.clip(points_new[:, 1], -0.5, 0.5)
                # Reset patience counter occasionally
                if no_improve > patience * 2:
                    no_improve = 0
                    # Try to recover some radius
                    r = max(current_r, r - r_step)
            else:
                # Reduce r target to allow recovery
                r = max(current_r, r - r_step * 0.5)
        
        points = points_new
        
        # Early stopping if we hit a very high precision or no change
        if best_r > 0.036612798904 + 1e-7:
             break

    return best_points

def construct():
    # Step 1: Generate initial hexagonal lattice
    points = _generate_dense_hex(N)
    
    # Step 2: Optimize using QP-style projected gradient descent
    # Use a larger number of iterations for the first pass to escape bad local minima
    points = _qp_optimize(points, max_iter=8000)
    
    # Step 3: Final polish with smaller steps (re-run to refine)
    points = _qp_optimize(points, max_iter=4000)
    
    return points
# EVOLVE-BLOCK-END
