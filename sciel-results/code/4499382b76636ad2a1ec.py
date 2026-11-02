"""Circle packing in perimeter-4 rectangle: Simulated Annealing Approach."""
import numpy as np
import time

N = 21


def _check_validity(cx, cy, r, W, H, tol=1e-9):
    """Check if the configuration is valid (no overlaps, within bounds)."""
    # Check bounds
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    
    # Check overlaps
    # Vectorized check for efficiency
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = cx_diff**2 + cy_diff**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    # We only need to check i < j
    # Create a mask for upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    # If any distance squared is less than (sum of radii)^2 - tol, it's an overlap
    # Note: using squared distances avoids sqrt
    if np.any(d2[mask] < (rs[mask] - tol)**2):
        return False
    return True


def _calc_penalty(cx, cy, r, W, H):
    """
    Calculate the penalty for constraint violations.
    This is used as a soft constraint in the SA objective.
    """
    penalty = 0.0
    
    # Boundary violations
    # Left
    viol = np.maximum(r - cx, 0)
    penalty += np.sum(viol**2)
    # Right
    viol = np.maximum(cx - (W - r), 0)
    penalty += np.sum(viol**2)
    # Bottom
    viol = np.maximum(r - cy, 0)
    penalty += np.sum(viol**2)
    # Top
    viol = np.maximum(cy - (H - r), 0)
    penalty += np.sum(viol**2)
    
    # Overlap violations
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = cx_diff**2 + cy_diff**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Calculate overlaps: max(0, r_i + r_j - d_ij)
    # To avoid sqrt, we can approximate or just use sqrt for the few overlapping pairs?
    # Actually, for SA, we need a smooth-ish penalty.
    # Let's compute overlaps properly.
    # d = sqrt(d2)
    # overlap = max(0, rs - d)
    # penalty += sum(overlap^2)
    
    # Optimization: Only compute sqrt for pairs that are potentially overlapping
    # i.e., d2 < rs^2
    potential_overlap_mask = mask & (d2 < rs**2)
    
    if np.any(potential_overlap_mask):
        # Get indices
        i_idx, j_idx = np.where(potential_overlap_mask)
        for i, j in zip(i_idx, j_idx):
            d = np.sqrt(d2[i, j])
            overlap = rs[i, j] - d
            if overlap > 0:
                penalty += overlap**2
                
    return penalty


def _objective(cx, cy, r, W, H, lambda_penalty=100.0):
    """
    Objective function to maximize.
    Maximize: sum(r) - lambda * penalty
    """
    return np.sum(r) - lambda_penalty * _calc_penalty(cx, cy, r, W, H)


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = -np.inf
    best_result = None
    
    # Initial Aspect Ratios to try
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2]
    
    for W_init in aspect_ratios:
        if time.time() > deadline - 20:
            break
            
        H_init = 2.0 - W_init
        if H_init <= 0.1:
            continue
            
        # Initialize circles in a grid
        # Determine grid dimensions
        # Try a few grid layouts
        layouts = [
            (3, 7), # 3 rows, 7 cols
            (7, 3),
            (4, 6), # 24 slots
            (6, 4),
            (5, 5), # 25 slots
        ]
        
        for rows, cols in layouts:
            if rows * cols < N:
                continue
                
            cell_w = W_init / cols
            cell_h = H_init / rows
            
            cx = np.zeros(N)
            cy = np.zeros(N)
            r = np.zeros(N)
            
            for i in range(N):
                row = i // cols
                col = i % cols
                cx[i] = (col + 0.5) * cell_w
                cy[i] = (row + 0.5) * cell_h
                # Start with a radius that fits in the cell with some margin
                r[i] = 0.4 * min(cell_w, cell_h)
            
            W = W_init
            H = H_init
            
            # Simulated Annealing Parameters
            T = 0.01  # Initial temperature
            T_min = 1e-6
            cooling_rate = 0.995
            max_iters = 20000
            lambda_penalty = 100.0
            
            current_obj = _objective(cx, cy, r, W, H, lambda_penalty)
            best_local_obj = current_obj
            best_local_result = (cx.copy(), cy.copy(), r.copy(), W, H)
            
            # SA Loop
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Generate a move
                move_type = np.random.random()
                
                cx_new = cx.copy()
                cy_new = cy.copy()
                r_new = r.copy()
                W_new = W
                H_new = H
                
                if move_type < 0.4:
                    # Move position of a random circle
                    idx = np.random.randint(N)
                    # Gaussian perturbation
                    sigma = 0.05 * T + 0.001
                    cx_new[idx] += np.random.normal(0, sigma)
                    cy_new[idx] += np.random.normal(0, sigma)
                    # Clamp to bounds (soft or hard? Hard is better for stability)
                    cx_new[idx] = np.clip(cx_new[idx], r_new[idx], W_new - r_new[idx])
                    cy_new[idx] = np.clip(cy_new[idx], r_new[idx], H_new - r_new[idx])
                    
                elif move_type < 0.8:
                    # Grow/Shrink a random circle
                    idx = np.random.randint(N)
                    # Small change in radius
                    delta_r = np.random.normal(0, 0.01 * T + 0.0001)
                    r_new[idx] += delta_r
                    # Ensure positive
                    r_new[idx] = max(r_new[idx], 1e-4)
                    # Clamp position if it now violates bounds
                    cx_new[idx] = np.clip(cx_new[idx], r_new[idx], W_new - r_new[idx])
                    cy_new[idx] = np.clip(cy_new[idx], r_new[idx], H_new - r_new[idx])
                    
                else:
                    # Adjust Aspect Ratio
                    # Change W slightly, H = 2 - W
                    delta_W = np.random.normal(0, 0.01 * T + 0.0001)
                    W_new = W + delta_W
                    W_new = np.clip(W_new, 0.4, 1.6)
                    H_new = 2.0 - W_new
                    # Rescale positions and radii? 
                    # No, just keep positions and let penalty handle it, 
                    # or scale them to fit? Scaling might be better for continuity.
                    # Let's just let the penalty handle the mismatch initially, 
                    # but this might be too harsh. 
                    # Better: Scale positions and radii proportionally to new W/H?
                    # Actually, just changing W/H while keeping positions fixed 
                    # might cause large violations. 
                    # Let's skip aspect ratio changes in the inner loop and 
                    # rely on the outer loop trying different W_inits.
                    # Or, do a small rescale.
                    # Let's just revert this move type to position for now to keep it simple.
                    cx_new = cx.copy()
                    cy_new = cy.copy()
                    r_new = r.copy()
                    W_new = W
                    H_new = H
                    idx = np.random.randint(N)
                    sigma = 0.05 * T + 0.001
                    cx_new[idx] += np.random.normal(0, sigma)
                    cy_new[idx] += np.random.normal(0, sigma)
                    cx_new[idx] = np.clip(cx_new[idx], r_new[idx], W_new - r_new[idx])
                    cy_new[idx] = np.clip(cy_new[idx], r_new[idx], H_new - r_new[idx])

                new_obj = _objective(cx_new, cy_new, r_new, W_new, H_new, lambda_penalty)
                delta_obj = new_obj - current_obj
                
                # Acceptance criterion
                if delta_obj > 0:
                    # Always accept improvements
                    cx, cy, r, W, H = cx_new, cy_new, r_new, W_new, H_new
                    current_obj = new_obj
                    
                    if current_obj > best_local_obj:
                        best_local_obj = current_obj
                        best_local_result = (cx.copy(), cy.copy(), r.copy(), W, H)
                else:
                    # Accept with probability exp(delta_obj / T)
                    if np.random.random() < np.exp(delta_obj / T):
                        cx, cy, r, W, H = cx_new, cy_new, r_new, W_new, H_new
                        current_obj = new_obj
                        
                # Cool down
                T *= cooling_rate
                
                # Check if we have a valid configuration to update global best
                # We check validity less frequently to save time, 
                # but we want to record valid states.
                if it % 10 == 0:
                    if _check_validity(cx, cy, r, W, H):
                        val = np.sum(r)
                        if val > best_local_obj: # Note: best_local_obj is based on penalty, 
                                                 # but we want to track raw sum(r) for valid configs
                             # Actually, if valid, penalty is 0, so obj = sum(r).
                             # So we can just use obj.
                             pass 
                        if val > best_val:
                            best_val = val
                            best_result = (cx.copy(), cy.copy(), r.copy(), W, H)

            # After SA, do a local polish on the best local result
            cx_ls, cy_ls, r_ls, W_ls, H_ls = best_local_result
            
            # Try to grow radii uniformly
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                r_trial = r_ls * 1.001
                if _check_validity(cx_ls, cy_ls, r_trial, W_ls, H_ls):
                    r_ls = r_trial
                else:
                    # Try small random moves to escape local optimum
                    idx = np.random.randint(N)
                    cx_t = cx_ls.copy()
                    cy_t = cy_ls.copy()
                    cx_t[idx] += np.random.normal(0, 0.01)
                    cy_t[idx] += np.random.normal(0, 0.01)
                    cx_t[idx] = np.clip(cx_t[idx], r_ls[idx], W_ls - r_ls[idx])
                    cy_t[idx] = np.clip(cy_t[idx], r_ls[idx], H_ls - r_ls[idx])
                    
                    if _check_validity(cx_t, cy_t, r_ls, W_ls, H_ls):
                        cx_ls = cx_t
                        cy_ls = cy_t
                        # Try to grow again
                        r_t2 = r_ls * 1.001
                        if _check_validity(cx_ls, cy_ls, r_t2, W_ls, H_ls):
                            r_ls = r_t2
                            
            # Update global best
            if _check_validity(cx_ls, cy_ls, r_ls, W_ls, H_ls):
                val = np.sum(r_ls)
                if val > best_val:
                    best_val = val
                    best_result = (cx_ls.copy(), cy_ls.copy(), r_ls.copy(), W_ls, H_ls)

    if best_result is None:
        # Fallback: Simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    cx, cy, r, W, H = best_result
    
    # Final safety check
    if not _check_validity(cx, cy, r, W, H):
        # Shrink until valid
        scale = 1.0
        while not _check_validity(cx, cy, r * scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        if not _check_validity(cx, cy, r, W, H):
             # Extreme fallback
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((cx, cy, r))
