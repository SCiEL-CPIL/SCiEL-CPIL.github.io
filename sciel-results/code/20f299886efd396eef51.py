import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    # Initial configuration: 3x3 grid + 1 small
    eps = 1e-6
    s = (1.0 / 3.0) - eps
    t = 1.0 - 3.0 * s
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    idx = 0
    for r in range(3):
        for c in range(3):
            cx = (c + 0.5) * s
            cy = (r + 0.5) * s
            centers[idx] = [cx, cy]
            sides[idx] = s
            idx += 1
            
    cx = 1.0 - 0.5 * t
    cy = 1.0 - 0.5 * t
    centers[idx] = [cx, cy]
    sides[idx] = t

    # Flatten for optimizer
    x0 = np.concatenate([centers.flatten(), sides])
    
    start_time = time.time()
    timeout = 590  # Safety margin
    
    def penalty_and_obj(x):
        centers_opt = x[:2*n].reshape(n, 2)
        sides_opt = x[2*n:]
        
        # Objective: Maximize sum of sides -> Minimize -sum
        obj_val = -np.sum(sides_opt)
        
        # Boundary Violations
        # cx - s/2 >= 0 => s/2 - cx <= 0
        # cx + s/2 <= 1 => cx + s/2 - 1 <= 0
        # Same for cy
        viol_b = 0.0
        for i in range(n):
            s_i = sides_opt[i]
            cx_i, cy_i = centers_opt[i]
            # Left
            v1 = max(0, s_i/2 - cx_i)
            # Right
            v2 = max(0, cx_i + s_i/2 - 1)
            # Bottom
            v3 = max(0, s_i/2 - cy_i)
            # Top
            v4 = max(0, cy_i + s_i/2 - 1)
            viol_b += v1**2 + v2**2 + v3**2 + v4**2
        
        # Overlap Violations
        viol_o = 0.0
        for i in range(n):
            for j in range(i+1, n):
                s_i, s_j = sides_opt[i], sides_opt[j]
                cx_i, cy_i = centers_opt[i]
                cx_j, cy_j = centers_opt[j]
                
                dx = abs(cx_i - cx_j)
                dy = abs(cy_i - cy_j)
                
                req_dx = (s_i + s_j) / 2.0
                req_dy = (s_i + s_j) / 2.0
                
                # Violation if dx < req_dx AND dy < req_dy
                vx = max(0, req_dx - dx)
                vy = max(0, req_dy - dy)
                
                # Product is 0 if either is 0. If both > 0, it's positive.
                viol_o += vx * vy
        
        # Total Penalty
        # Scale penalty heavily to enforce constraints
        pen = 100.0 * (viol_b + viol_o)
        
        return obj_val + pen

    best_x = x0
    best_val = penalty_and_obj(x0)
    
    # Run Nelder-Mead
    try:
        res = minimize(penalty_and_obj, x0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        if time.time() - start_time < timeout:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
    except Exception:
        pass

    # Parse best_x
    centers_best = best_x[:2*n].reshape(n, 2)
    sides_best = best_x[2*n:]
    
    # Ensure sides are positive
    sides_best = np.maximum(sides_best, 1e-9)
    # Re-parse to be safe
    centers_best = best_x[:2*n].reshape(n, 2)
    sides_best = best_x[2*n:]
    
    # Final validity check and repair
    # If invalid, shrink slightly
    def is_valid(c, s):
        tol = 1e-7
        # Bounds
        if np.any(c - s/2 < -tol) or np.any(c + s/2 > 1 + tol):
            return False
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                t_req = (s[i] + s[j]) / 2.0 - tol
                if abs(c[i,0] - c[j,0]) < t_req and abs(c[i,1] - c[j,1]) < t_req:
                    return False
        return True

    if not is_valid(centers_best, sides_best):
        # Try to repair by shrinking
        # This is a simple fallback
        scale = 0.99
        for _ in range(10):
            sides_new = sides_best * scale
            centers_new = centers_best # Centers might need adjustment if we shrink, but if they were overlapping, shrinking helps
            if is_valid(centers_new, sides_new):
                centers_best = centers_new
                sides_best = sides_new
                break
            scale *= 0.95
        else:
            # If still invalid, return the original seed config which is known valid
            return centers, sides
            
    return centers_best, sides_best
