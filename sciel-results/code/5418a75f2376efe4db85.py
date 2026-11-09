import numpy as np
import time
from scipy.optimize import differential_evolution

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 590.0  # Safety margin for 600s limit
    
    # Variables: 10 * (cx, cy, s) = 30 variables
    # Bounds: cx, cy in [0, 1], s in [0.01, 1.0]
    # Actually, s can be up to 1.
    
    def penalty_and_objective(params):
        # Unpack params
        # params shape (30,)
        # 0:2 cx, 2:4 cy, 4:6 s ... etc?
        # Better: reshape to (10, 3)
        p = params.reshape(n, 3)
        cx = p[:, 0]
        cy = p[:, 1]
        s = p[:, 2]
        
        # Objective: maximize sum(s) -> minimize -sum(s)
        obj = -np.sum(s)
        
        # Constraints
        # 1. Inside unit square
        # min_x = cx - s/2 >= 0
        # max_x = cx + s/2 <= 1
        # min_y = cy - s/2 >= 0
        # max_y = cy + s/2 <= 1
        
        min_x = cx - s/2
        max_x = cx + s/2
        min_y = cy - s/2
        max_y = cy + s/2
        
        # Violations
        v_x_min = np.clip(-min_x, 0, None) # >0 if violation
        v_x_max = np.clip(max_x - 1, 0, None)
        v_y_min = np.clip(-min_y, 0, None)
        v_y_max = np.clip(max_y - 1, 0, None)
        
        boundary_violation = np.sum(v_x_min) + np.sum(v_x_max) + np.sum(v_y_min) + np.sum(v_y_max)
        
        # 2. Overlap
        # For each pair i, j:
        # overlap_x = (s_i + s_j)/2 - |cx_i - cx_j|
        # overlap_y = (s_i + s_j)/2 - |cy_i - cy_j|
        # If overlap_x > 0 AND overlap_y > 0, they overlap.
        # Penalty = sum of max(0, overlap_x) * max(0, overlap_y)? Or just sum of overlaps?
        # A common penalty is the area of intersection or just the extent.
        # Let's use the product of positive overlaps to penalize actual intersection area roughly.
        # Or simply: if they overlap, add a large penalty.
        # Using a smooth penalty:
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i+1, n):
                ox = (s[i] + s[j])/2.0 - abs(cx[i] - cx[j])
                oy = (s[i] + s[j])/2.0 - abs(cy[i] - cy[j])
                if ox > 0 and oy > 0:
                    # Penalize by the overlap area or just a constant + overlap amount
                    # Using overlap area helps guide the optimizer
                    overlap_penalty += ox * oy
                    
        # Total penalty
        # Scale penalty to be larger than any possible objective value
        # Max possible sum of sides is ~3.16.
        # Penalty weight should be high.
        total_penalty = boundary_violation * 100.0 + overlap_penalty * 100.0
        
        return obj + total_penalty

    # Bounds for DE
    # cx, cy in [0, 1]
    # s in [0.01, 1.0] (avoid 0 to prevent degenerate solutions, though 0 is allowed, >0 is required)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s

    best_result = None
    best_score = -np.inf
    
    # Run DE for a few seconds or iterations
    # maxiter is tricky with time limit.
    # We can run DE with a moderate maxiter and check time.
    
    # Initial guess: random
    # DE is stochastic.
    
    try:
        # Use a wrapper to check time? DE doesn't support callbacks for time easily in older SciPy.
        # We can just run it with a reasonable maxiter and hope it finishes.
        # 30 variables. Popsize 10-20. Maxiter 200-500.
        # This should be fast.
        
        res = differential_evolution(
            penalty_and_objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=15,
            tol=0.01,
            polish=True
        )
        
        if res.success or res.fun < best_score:
            best_result = res.x
            best_score = res.fun
            
    except Exception:
        pass
        
    # If time remains, try another seed or more iterations
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        try:
            res2 = differential_evolution(
                penalty_and_objective, 
                bounds, 
                seed=123, 
                maxiter=300, 
                popsize=15,
                tol=0.01,
                polish=True
            )
            if res2.success or res2.fun < best_score:
                best_result = res2.x
                best_score = res2.fun
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to grid
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        return centers, sides

    # Extract best result
    p = best_result.reshape(n, 3)
    cx = p[:, 0]
    cy = p[:, 1]
    s = p[:, 2]
    
    centers = np.stack([cx, cy], axis=1)
    sides = s
    
    # Final validation and clamping to ensure validity
    # If there are tiny violations, we might need to adjust.
    # But DE with penalty should be clean.
    # To be safe, we can do a final check and if invalid, fallback.
    
    # Check validity
    valid = True
    for i in range(n):
        if s[i] <= 0:
            valid = False
            break
        if cx[i] - s[i]/2 < -1e-7 or cx[i] + s[i]/2 > 1 + 1e-7:
            valid = False
            break
        if cy[i] - s[i]/2 < -1e-7 or cy[i] + s[i]/2 > 1 + 1e-7:
            valid = False
            break
            
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2.0 - 1e-7
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback to grid
        k = 4
        side = 1.0/k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1)
        sides = np.full(n, side)
        
    return centers, sides
