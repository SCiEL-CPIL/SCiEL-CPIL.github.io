"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Perimeter 4 => w + h = 2.
    # We parameterize the rectangle by aspect ratio ar = w/h.
    # h = 2 / (1 + ar), w = 2 * ar / (1 + ar).
    
    def get_wh(ar):
        h = 2.0 / (1.0 + ar)
        w = 2.0 * ar / (1.0 + ar)
        return w, h

    def objective(params):
        """
        Minimize: -sum(radii) + penalty(overlaps) + penalty(boundary)
        params: [ar, x_1, y_1, r_1, ..., x_21, y_21, r_21]
        """
        ar = params[0]
        # Clamp ar to reasonable bounds to avoid degenerate rectangles
        ar = np.clip(ar, 0.5, 3.0)
        w, h = get_wh(ar)
        
        circles = params[1:].reshape(N, 3)
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        # Ensure positive radii
        r = np.maximum(r, 1e-6)
        
        # 1. Boundary Penalty
        # We want x in [r, w-r] and y in [r, h-r]
        # Penalty if x < r or x > w-r
        pen_x = np.maximum(0, r - x) + np.maximum(0, x - (w - r))
        pen_y = np.maximum(0, r - y) + np.maximum(0, y - (h - r))
        boundary_pen = np.sum(pen_x**2 + pen_y**2)
        
        # 2. Overlap Penalty
        # We want dist(i,j) >= r_i + r_j
        # Penalty if dist < r_i + r_j
        # Vectorized computation for all pairs
        # To save memory and time, we do it in a loop but vectorized over the vector parts
        overlap_pen = 0.0
        for i in range(N):
            dx = x[i] - x[i+1:]
            dy = y[i] - y[i+1:]
            dist_sq = dx**2 + dy**2
            min_dist = r[i] + r[i+1:]
            
            # Only care if dist_sq < min_dist^2
            # We use a smooth approximation for the penalty to help gradients
            # penalty = max(0, min_dist - dist)^2
            # To avoid sqrt in the inner loop if possible, or just use it carefully
            # Since N=21, O(N^2) is small (210 pairs). Sqrt is fine.
            
            dist = np.sqrt(dist_sq + 1e-12)
            overlap = np.maximum(0, min_dist - dist)
            overlap_pen += np.sum(overlap**2)
            
        # 3. Objective
        # We want to MAXIMIZE sum(r), so MINIMIZE -sum(r)
        # Weight the penalties heavily to enforce constraints
        sum_r = np.sum(r)
        
        # Adaptive penalty weight? 
        # Start with a high weight to get a valid config, then lower it?
        # For L-BFGS, a fixed high weight is often safer if we start from a valid state.
        # However, if we start from an invalid state, we need to push out overlaps.
        
        penalty_weight = 50.0
        score = -sum_r + penalty_weight * (boundary_pen + overlap_pen)
        
        return score

    def objective_grad(params):
        """
        Numerical gradient is slow but robust for this small N.
        L-BFGS-B can handle numerical gradients, but providing analytic is better.
        Given time constraints, we will use numerical gradient (finite difference) 
        provided by scipy if we don't implement analytic.
        Actually, implementing analytic gradient for max(0, ...) is tricky due to non-differentiability.
        We will rely on scipy's numerical gradient for simplicity and robustness,
        or use a smooth penalty like exp() or sigmoid.
        
        Let's use a smooth penalty to make gradients well-defined.
        penalty = sum( exp(k * (min_dist - dist)) - 1 ) for overlaps?
        Or just stick to L-BFGS-B with finite differences. N=21, dim=64. 
        64 evaluations per gradient step. 1000 steps = 64000 evals. 
        Each eval is O(N^2) ~ 200 ops. 12.8M ops. Fast enough.
        """
        return None # None means use numerical gradient

    best_circles = None
    best_ar = 1.0
    best_value = -np.inf
    
    # Multi-start strategy
    # Try different initial aspect ratios and grid configurations
    
    aspect_ratios = [0.7, 1.0, 1.4, 2.0, 2.5]
    
    for ar_init in aspect_ratios:
        if time.time() > deadline - 10:
            break
            
        w, h = get_wh(ar_init)
        
        # Initialize with a grid
        # 3x7 grid is good for ar ~ 2.33 (w=1.33, h=0.66). 
        # For ar=1.0 (1x1), 5x5 grid (25 slots, use 21) or 3x7?
        # Let's just place them in a grid that fits.
        
        # Heuristic: try to fit 3 rows of 7
        rows = 3
        cols = 7
        # If aspect ratio is close to 1, maybe 4x5 or 5x5?
        # Let's just use a standard grid init and let optimizer fix it.
        
        init_params = np.zeros(1 + N*3)
        init_params[0] = ar_init
        
        # Grid positions
        # Margin to ensure we start valid-ish
        margin = 0.05
        w_eff = w - 2*margin
        h_eff = h - 2*margin
        
        if w_eff <= 0 or h_eff <= 0:
            continue
            
        # Determine grid dimensions based on aspect ratio to keep cells square-ish
        # If w > h, more cols.
        if w > h:
            # 3 rows, 7 cols
            r_idx = 0
            for i in range(3):
                for j in range(7):
                    # Center of cell
                    cx = margin + (j + 0.5) * (w_eff / 7)
                    cy = margin + (i + 0.5) * (h_eff / 3)
                    # Max radius is half min(cell_w, cell_h)
                    cell_w = w_eff / 7
                    cell_h = h_eff / 3
                    r_init = 0.45 * min(cell_w, cell_h)
                    
                    idx = 1 + r_idx*3
                    init_params[idx] = cx
                    init_params[idx+1] = cy
                    init_params[idx+2] = r_init
                    r_idx += 1
        else:
            # 7 rows, 3 cols
            r_idx = 0
            for i in range(7):
                for j in range(3):
                    cx = margin + (j + 0.5) * (w_eff / 3)
                    cy = margin + (i + 0.5) * (h_eff / 7)
                    cell_w = w_eff / 3
                    cell_h = h_eff / 7
                    r_init = 0.45 * min(cell_w, cell_h)
                    
                    idx = 1 + r_idx*3
                    init_params[idx] = cx
                    init_params[idx+1] = cy
                    init_params[idx+2] = r_init
                    r_idx += 1

        # Add some noise to break symmetry
        init_params[1:] += np.random.normal(0, 0.01, size=N*3)
        
        # Bounds for L-BFGS-B
        # ar: [0.5, 3.0]
        # x, y, r: [0, 2] (loose bounds, penalty handles tightness)
        # Actually, tighter bounds help.
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        # Since w,h change with ar, bounds are dynamic. L-BFGS-B requires static bounds.
        # So we use loose bounds and rely on penalty.
        # Or we can re-parameterize to unit square and scale?
        # Let's stick to loose bounds [0, 2] for x,y,r and [0.5, 3.0] for ar.
        
        bounds = [(0.5, 3.0)]
        for _ in range(N):
            bounds.append((0.0, 2.0)) # x
            bounds.append((0.0, 2.0)) # y
            bounds.append((0.0, 1.0)) # r (max radius can't be > 1 if w+h=2)
            
        # Run optimization
        try:
            res = minimize(
                objective, 
                init_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_value: # Note: res.fun is negative sum_r + pen
                # Extract result
                ar_opt = res.x[0]
                circles_opt = res.x[1:].reshape(N, 3)
                
                # Check validity
                w_opt, h_opt = get_wh(ar_opt)
                valid = True
                sum_r = 0
                for i in range(N):
                    x, y, r = circles_opt[i]
                    if x < r - 1e-7 or x > w_opt - r + 1e-7 or y < r - 1e-7 or y > h_opt - r + 1e-7:
                        valid = False
                        break
                    sum_r += r
                    
                if valid:
                    for i in range(N):
                        for j in range(i+1, N):
                            dx = circles_opt[i,0] - circles_opt[j,0]
                            dy = circles_opt[i,1] - circles_opt[j,1]
                            dist_sq = dx*dx + dy*dy
                            min_dist = circles_opt[i,2] + circles_opt[j,2]
                            if dist_sq < min_dist*min_dist - 1e-7:
                                valid = False
                                break
                        if not valid: break
                
                if valid:
                    # The objective was -sum_r + penalty. 
                    # If valid, penalty should be ~0.
                    # So value is roughly sum_r.
                    # Let's calculate exact sum_r for comparison
                    val = np.sum(circles_opt[:, 2])
                    if val > best_value:
                        best_value = val
                        best_circles = circles_opt.copy()
                        best_ar = ar_opt
                        
        except Exception:
            pass

    if best_circles is not None:
        return best_circles
    else:
        # Fallback: Return a valid small grid
        # 3x7 grid in 1.4 x 0.6
        w, h = 1.4, 0.6
        fallback = np.zeros((N, 3))
        idx = 0
        for i in range(3):
            for j in range(7):
                cx = (j + 0.5) * (w / 7)
                cy = (i + 0.5) * (h / 3)
                r = 0.4 * min(w/7, h/3)
                fallback[idx] = [cx, cy, r]
                idx += 1
        return fallback

# EVOLVE-BLOCK-END
