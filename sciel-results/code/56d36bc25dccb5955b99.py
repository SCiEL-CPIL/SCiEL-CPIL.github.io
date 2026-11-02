"""Circle packing in perimeter-4 rectangle: Physics-based Relaxation."""
import numpy as np
import time
import random

N = 21
_UPPER_MASK = np.triu(np.ones((N, N), dtype=bool), k=1)
_UPPER_IDX = np.where(_UPPER_MASK)

def _check_valid(cx, cy, r, W, H, tol=1e-9):
    """Check if the configuration is valid."""
    if W <= 0 or H <= 0:
        return False
    if np.any(cx < r - tol) or np.any(cx > W - r + tol):
        return False
    if np.any(cy < r - tol) or np.any(cy > H - r + tol):
        return False
    # Vectorized overlap check using precomputed indices
    dx = cx[_UPPER_IDX[0]] - cx[_UPPER_IDX[1]]
    dy = cy[_UPPER_IDX[0]] - cy[_UPPER_IDX[1]]
    d2 = dx**2 + dy**2
    rs = r[_UPPER_IDX[0]] + r[_UPPER_IDX[1]]
    if np.any(d2 < (rs - tol)**2):
        return False
    return True

def _relax_physics(cx, cy, r, W, H, step, iterations=1000, time_limit=None, start_time=None):
    """
    Physics-based relaxation with vectorized force computation.
    W + H = 2 is fixed. We vary W.
    """
    if start_time is None:
        start_time = time.time()
    
    best_val = -np.inf
    best_state = (cx.copy(), cy.copy(), r.copy(), W, H)
    
    current_r = r.copy()
    current_cx = cx.copy()
    current_cy = cy.copy()
    current_W = W
    current_H = H
    
    # Adaptive parameters
    lr_move = 0.05
    lr_grow = 0.002
    lr_aspect = 0.005
    
    for it in range(iterations):
        if time.time() - start_time > (time_limit if time_limit else 110.0):
            break
            
        # 1. Resolve overlaps using vectorized forces
        for _ in range(8):
            # Compute pairwise differences
            dx = current_cx[:, np.newaxis] - current_cx[np.newaxis, :]
            dy = current_cy[:, np.newaxis] - current_cy[np.newaxis, :]
            d2 = dx**2 + dy**2
            rs = current_r[:, np.newaxis] + current_r[np.newaxis, :]
            
            # Only consider upper triangle pairs
            d2_upper = d2[_UPPER_IDX]
            rs_upper = rs[_UPPER_IDX]
            dx_upper = dx[_UPPER_IDX]
            dy_upper = dy[_UPPER_IDX]
            
            # Find overlapping pairs
            overlap_mask = d2_upper < rs_upper**2
            if not np.any(overlap_mask):
                break
            
            # Compute forces for overlapping pairs
            d_upper = np.sqrt(d2_upper[overlap_mask] + 1e-12)
            overlap_amount = rs_upper[overlap_mask] - d_upper
            force_mag = overlap_amount * 0.5
            
            # Directions
            fx = force_mag * dx_upper[overlap_mask] / d_upper
            fy = force_mag * dy_upper[overlap_mask] / d_upper
            
            # Apply forces using scatter
            forces_x = np.zeros(N)
            forces_y = np.zeros(N)
            i_idx = _UPPER_IDX[0][overlap_mask]
            j_idx = _UPPER_IDX[1][overlap_mask]
            np.add.at(forces_x, i_idx, fx)
            np.add.at(forces_y, i_idx, fy)
            np.add.at(forces_x, j_idx, -fx)
            np.add.at(forces_y, j_idx, -fy)
            
            # Apply forces
            current_cx += forces_x * lr_move
            current_cy += forces_y * lr_move
            
            # Project back to bounds
            current_cx = np.clip(current_cx, current_r, current_W - current_r)
            current_cy = np.clip(current_cy, current_r, current_H - current_r)
            
        # 2. Check if we can grow
        growth_factor = 1.0 + lr_grow
        new_r = current_r * growth_factor
        
        # Check if new radii are valid with current positions
        valid_growth = True
        # Bounds check for growth
        if np.any(new_r > current_cx) or np.any(new_r > current_W - current_cx):
             valid_growth = False
        if np.any(new_r > current_cy) or np.any(new_r > current_H - current_cy):
             valid_growth = False
             
        if valid_growth:
             # Check overlaps with new radii using precomputed indices
             dx = current_cx[_UPPER_IDX[0]] - current_cx[_UPPER_IDX[1]]
             dy = current_cy[_UPPER_IDX[0]] - current_cy[_UPPER_IDX[1]]
             d2 = dx**2 + dy**2
             rs = new_r[_UPPER_IDX[0]] + new_r[_UPPER_IDX[1]]
             if np.any(d2 < (rs - 1e-9)**2):
                 valid_growth = False
                 
        if valid_growth:
            current_r = new_r
            val = np.sum(current_r)
            if val > best_val:
                best_val = val
                best_state = (current_cx.copy(), current_cy.copy(), current_r.copy(), current_W, current_H)
            
            # Decay growth rate slightly to avoid instability
            lr_grow *= 0.9995
        else:
            # If can't grow, reduce growth rate
            lr_grow *= 0.95
            # Jiggle to escape local minima
            if it % 30 == 0:
                jiggle = 0.02 * (1.0 - it/iterations)
                current_cx += np.random.normal(0, jiggle, N)
                current_cy += np.random.normal(0, jiggle, N)
                current_cx = np.clip(current_cx, current_r, current_W - current_r)
                current_cy = np.clip(current_cy, current_r, current_H - current_r)
                
        # 3. Adjust Aspect Ratio every 10 iterations
        if it % 10 == 0:
            dist_left = current_cx - current_r
            dist_right = current_W - current_cx - current_r
            dist_bottom = current_cy - current_r
            dist_top = current_H - current_cy - current_r
            
            avg_dist_x = (np.mean(dist_left) + np.mean(dist_right)) / 2.0
            avg_dist_y = (np.mean(dist_bottom) + np.mean(dist_top)) / 2.0
            
            # Only adjust if there's a meaningful difference
            if abs(avg_dist_x - avg_dist_y) > 0.01:
                if avg_dist_x < avg_dist_y:
                    current_W += lr_aspect
                else:
                    current_W -= lr_aspect
                
                current_W = np.clip(current_W, 0.5, 1.5)
                current_H = 2.0 - current_W
                
                # Re-project positions to new bounds
                current_cx = np.clip(current_cx, current_r, current_W - current_r)
                current_cy = np.clip(current_cy, current_r, current_H - current_r)

    return best_state

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = 0.0
    best_result = None
    
    # Try multiple initial configurations
    configs = []
    
    # Grid layouts with varying aspect ratios
    grid_dims = [(3, 7), (7, 3), (4, 6), (6, 4), (5, 5)]
    aspect_ratios = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    for rows, cols in grid_dims:
        if rows * cols < N:
            continue
        for W_init in aspect_ratios:
            H_init = 2.0 - W_init
            cx = np.zeros(N)
            cy = np.zeros(N)
            r = np.zeros(N)
            cell_w = W_init / cols
            cell_h = H_init / rows
            for i in range(N):
                row = i // cols
                col = i % cols
                cx[i] = (col + 0.5) * cell_w
                cy[i] = (row + 0.5) * cell_h
                r[i] = min(cell_w, cell_h) * 0.4
            configs.append((cx, cy, r, W_init, H_init))
        
    # Random initializations with different aspect ratios
    for W_init in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        H_init = 2.0 - W_init
        cx = np.random.uniform(0.05, W_init - 0.05, N)
        cy = np.random.uniform(0.05, H_init - 0.05, N)
        r = np.full(N, 0.08)
        configs.append((cx, cy, r, W_init, H_init))
        
    # Shuffle configs to try diverse ones early
    random.shuffle(configs)
    
    for cx, cy, r, W, H in configs:
        if time.time() > deadline - 15:
            break
            
        # Run physics relaxation
        res_cx, res_cy, res_r, res_W, res_H = _relax_physics(
            cx, cy, r, W, H, 
            step=1.0, 
            iterations=3000, 
            time_limit=8.0, 
            start_time=time.time()
        )
        
        if _check_valid(res_cx, res_cy, res_r, res_W, res_H):
            val = np.sum(res_r)
            if val > best_val:
                best_val = val
                best_result = (res_cx, res_cy, res_r, res_W, res_H)
                
    if best_result is None:
        # Fallback: simple grid in square
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    cx, cy, r, W, H = best_result
    
    # Final Polish: Local Search
    for _ in range(1000):
        if time.time() > deadline - 2:
            break
        # Try growing radii
        r_trial = r * 1.0003
        if _check_valid(cx, cy, r_trial, W, H):
            r = r_trial
            continue
        # Try moving a random circle
        idx = np.random.randint(N)
        cx_trial = cx.copy()
        cy_trial = cy.copy()
        cx_trial[idx] += np.random.normal(0, 0.008)
        cy_trial[idx] += np.random.normal(0, 0.008)
        cx_trial[idx] = np.clip(cx_trial[idx], r[idx], W - r[idx])
        cy_trial[idx] = np.clip(cy_trial[idx], r[idx], H - r[idx])
        
        if _check_valid(cx_trial, cy_trial, r, W, H):
            cx = cx_trial
            cy = cy_trial
            r_trial = r * 1.0003
            if _check_valid(cx, cy, r_trial, W, H):
                r = r_trial
        # Try adjusting aspect ratio slightly
        W_trial = W + np.random.normal(0, 0.015)
        W_trial = np.clip(W_trial, 0.4, 1.6)
        H_trial = 2.0 - W_trial
        if _check_valid(cx, cy, r, W_trial, H_trial):
            r_trial2 = r * 1.0003
            if _check_valid(cx, cy, r_trial2, W_trial, H_trial):
                W = W_trial
                H = H_trial
                r = r_trial2

    # Final safety check
    if not _check_valid(cx, cy, r, W, H):
        scale = 1.0
        while not _check_valid(cx, cy, r * scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        if not _check_valid(cx, cy, r, W, H):
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((cx, cy, r))
