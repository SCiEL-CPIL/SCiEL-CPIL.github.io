"""Circle packing in perimeter-4 rectangle: Gradient Descent with Physics-based Relaxation."""
import numpy as np
import time

N = 21


def _compute_forces_and_gradient(cx, cy, r, W, H, lambda_overlap=50.0, lambda_bound=50.0):
    """
    Compute the gradient of the penalty function.
    We want to minimize:
    P = -sum(r) + lambda_o * sum(overlap^2) + lambda_b * sum(boundary_violation^2)
    
    Actually, we want to MAXIMIZE sum(r) while keeping violations ~ 0.
    So we can view this as:
    Step = -grad(Penalty) + alpha * grad(SumR)
    But SumR gradient is just 1 for all radii.
    
    Let's define Energy E = sum(overlaps^2) + sum(boundary_violations^2)
    We want to minimize E while increasing sum(r).
    
    Gradient of E wrt positions:
    For overlap i,j: term is max(0, r_i+r_j - d_ij)^2.
    d_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)
    If d_ij < r_i+r_j:
      overlap = r_i+r_j - d_ij
      d(overlap^2)/d(x_i) = 2*overlap * d(overlap)/d(x_i)
      d(overlap)/d(x_i) = -d(d_ij)/d(x_i) = -(x_i-x_j)/d_ij
      So grad_x_i += 2*overlap * (x_i-x_j)/d_ij
      grad_x_j += 2*overlap * (x_j-x_i)/d_ij = -grad_x_i
      
    Gradient of E wrt radii:
    For overlap i,j: term is max(0, r_i+r_j - d_ij)^2.
    d(overlap^2)/d(r_i) = 2*overlap * 1
    So grad_r_i += 2*overlap
    
    Boundary:
    If x_i < r_i: viol = r_i - x_i. term = viol^2.
    d(viol^2)/d(x_i) = 2*viol * (-1) = -2*viol
    d(viol^2)/d(r_i) = 2*viol * (1) = 2*viol
    """
    gx = np.zeros(N)
    gy = np.zeros(N)
    gr = np.zeros(N)
    
    # Overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx * dx + dy * dy
            d = np.sqrt(max(d2, 1e-12))
            rs = r[i] + r[j]
            if d < rs:
                overlap = rs - d
                # Gradient contribution
                # factor = 2 * overlap / d
                factor = 2.0 * overlap / d
                gx[i] += factor * dx
                gx[j] -= factor * dx
                gy[i] += factor * dy
                gy[j] -= factor * dy
                gr[i] += 2.0 * overlap
                gr[j] += 2.0 * overlap
    
    # Boundaries
    for i in range(N):
        # Left
        if cx[i] < r[i]:
            viol = r[i] - cx[i]
            gx[i] -= 2.0 * viol
            gr[i] += 2.0 * viol
        # Right
        if cx[i] > W - r[i]:
            viol = cx[i] - (W - r[i])
            gx[i] += 2.0 * viol
            gr[i] += 2.0 * viol
        # Bottom
        if cy[i] < r[i]:
            viol = r[i] - cy[i]
            gy[i] -= 2.0 * viol
            gr[i] += 2.0 * viol
        # Top
        if cy[i] > H - r[i]:
            viol = cy[i] - (H - r[i])
            gy[i] += 2.0 * viol
            gr[i] += 2.0 * viol
            
    return gx, gy, gr


def _relax_iteration(cx, cy, r, W, H, lr_pos, lr_rad, lr_aspect, lambda_obj=1.0):
    """
    Perform one gradient descent step to minimize violations, 
    and a step to maximize radii.
    """
    gx, gy, gr = _compute_forces_and_gradient(cx, cy, r, W, H)
    
    # Gradient of Objective: We want to maximize sum(r), so gradient is +1 for all r.
    # We combine: 
    # x_new = x - lr_pos * gx
    # r_new = r - lr_rad * (gr - lambda_obj)  <-- Note: minimize (violations - lambda*sum_r)
    # Wait, if we minimize E - lambda*sum_r, then grad_r = grad_E - lambda.
    # If lambda is large, we prioritize growing radii even if it causes some violation (which we then fix in next iter).
    
    cx_new = cx - lr_pos * gx
    cy_new = cy - lr_pos * gy
    r_new = r - lr_rad * (gr - lambda_obj)
    
    # Clamp radii to positive
    r_new = np.maximum(r_new, 1e-4)
    
    # Clamp positions to bounds (hard projection to ensure we don't fly out)
    cx_new = np.clip(cx_new, r_new, W - r_new)
    cy_new = np.clip(cy_new, r_new, H - r_new)
    
    # Adjust Aspect Ratio
    # Measure "tightness" in x and y
    # If circles are hitting left/right walls more than bottom/top, widen W.
    wall_x = np.sum(np.maximum(r - cx, 0)) + np.sum(np.maximum(cx - (W - r), 0))
    wall_y = np.sum(np.maximum(r - cy, 0)) + np.sum(np.maximum(cy - (H - r), 0))
    
    # Simple heuristic: if wall_x > wall_y, increase W (decrease H)
    # W is in [0.5, 1.5] approx (since perimeter 4, W+H=2)
    W_new = W + lr_aspect * (wall_y - wall_x)
    W_new = np.clip(W_new, 0.4, 1.6)
    H_new = 2.0 - W_new
    
    return cx_new, cy_new, r_new, W_new, H_new


def _is_valid_fast(cx, cy, r, W, H, tol=1e-8):
    """Fast validity check."""
    if W <= 0.1 or H <= 0.1:
        return False
    if np.any(cx - r < -tol) or np.any(cy - r < -tol):
        return False
    if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
        return False
    # Vectorized overlap check
    # This is O(N^2) but vectorized
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = cx_diff**2 + cy_diff**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    # We only care about upper triangle
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(d2[mask] < (rs[mask] - tol)**2):
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_val = 0.0
    best_result = None
    
    # Try multiple aspect ratios and initial configurations
    aspect_ratios = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    
    for W_init in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        H_init = 2.0 - W_init
        if H_init <= 0.1:
            continue
            
        # Initialize grid
        # Calculate optimal grid dimensions for N circles in W x H
        # Try a few grid layouts
        layouts = [
            (3, 7), # 3x7 = 21
            (7, 3),
            (4, 6), # 24 slots, use 21
            (6, 4),
            (5, 5), # 25 slots, use 21
        ]
        
        for rows, cols in layouts:
            if rows * cols < N:
                continue
                
            # Calculate cell size
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
                r[i] = min(cell_w, cell_h) * 0.45 # Start small to allow movement
            
            W = W_init
            H = H_init
            
            # Relaxation Loop
            lr_pos = 0.05
            lr_rad = 0.01
            lr_aspect = 0.01
            lambda_obj = 10.0 # Aggressiveness of radius growth
            
            # Adaptive learning rates
            current_violation = 1e9
            stagnation_count = 0
            
            for it in range(2000):
                if time.time() > deadline - 5:
                    break
                    
                cx, cy, r, W, H = _relax_iteration(cx, cy, r, W, H, lr_pos, lr_rad, lr_aspect, lambda_obj)
                
                # Check validity and update best
                if _is_valid_fast(cx, cy, r, W, H):
                    val = np.sum(r)
                    if val > best_val:
                        best_val = val
                        best_result = (cx.copy(), cy.copy(), r.copy(), W, H)
                    
                    # If valid, we can try to grow radii more aggressively in next steps
                    # by increasing lambda_obj slightly or resetting learning rates
                    if val > current_violation - 1e-4: # Pseudo-violation measure
                        pass # Just tracking best
                    
                    # Decay learning rates slowly
                    lr_pos *= 0.999
                    lr_rad *= 0.999
                    lambda_obj *= 1.001 # Slightly more aggressive growth over time
                    
                else:
                    # If invalid, reduce growth aggressiveness and increase position adjustment
                    lambda_obj *= 0.95
                    lr_pos *= 1.05
                    lr_rad *= 0.95
                    
                # Prevent divergence
                if not np.all(np.isfinite(cx)) or not np.all(np.isfinite(r)):
                    break

    # Final Polish: Local Search on best result
    if best_result is not None and time.time() < deadline - 10:
        cx_ls, cy_ls, r_ls, W_ls, H_ls = best_result
        # Simple local search to squeeze out more
        for _ in range(5000):
            if time.time() > deadline - 2:
                break
            # Try growing radii
            r_trial = r_ls * 1.001
            if _is_valid_fast(cx_ls, cy_ls, r_trial, W_ls, H_ls):
                r_ls = r_trial
            else:
                # Try moving a random circle
                idx = np.random.randint(N)
                cx_trial = cx_ls.copy()
                cy_trial = cy_ls.copy()
                cx_trial[idx] += np.random.normal(0, 0.01)
                cy_trial[idx] += np.random.normal(0, 0.01)
                cx_trial[idx] = np.clip(cx_trial[idx], r_ls[idx], W_ls - r_ls[idx])
                cy_trial[idx] = np.clip(cy_trial[idx], r_ls[idx], H_ls - r_ls[idx])
                
                # Check if this move allows growth
                if _is_valid_fast(cx_trial, cy_trial, r_ls, W_ls, H_ls):
                    # Try to grow
                    r_trial2 = r_ls * 1.002
                    if _is_valid_fast(cx_trial, cy_trial, r_trial2, W_ls, H_ls):
                        cx_ls = cx_trial
                        cy_ls = cy_trial
                        r_ls = r_trial2
                        val = np.sum(r_ls)
                        if val > best_val:
                            best_val = val
                            best_result = (cx_ls.copy(), cy_ls.copy(), r_ls.copy(), W_ls, H_ls)
                    else:
                        cx_ls = cx_trial
                        cy_ls = cy_trial

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    cx, cy, r, W, H = best_result
    
    # Final safety check
    if not _is_valid_fast(cx, cy, r, W, H):
        # Shrink until valid
        scale = 1.0
        while not _is_valid_fast(cx, cy, r * scale, W, H) and scale > 0.1:
            scale *= 0.95
        r = r * scale
        if not _is_valid_fast(cx, cy, r, W, H):
             # Extreme fallback
             side = int(np.ceil(np.sqrt(N)))
             cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                      for i in range(side) for j in range(side)]
             return np.array(cells[:N])

    return np.column_stack((cx, cy, r))
