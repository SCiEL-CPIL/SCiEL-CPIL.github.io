import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    # Objective function to minimize: -sum(sides) + penalty
    def objective(params):
        # params shape: (30,)
        # Reshape to (10, 3): cx, cy, s
        p = params.reshape(n, 3)
        cx = p[:, 0]
        cy = p[:, 1]
        s = p[:, 2]
        
        # Ensure sides are positive
        # If s is very small, it's fine, but we want to maximize sum.
        
        penalty = 0.0
        
        # 1. Boundary constraints
        # Square i is [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2]
        # Must be in [-tol, 1+tol]
        left = cx - s / 2.0
        right = cx + s / 2.0
        bottom = cy - s / 2.0
        top = cy + s / 2.0
        
        # Penalty for going out of bounds
        # Use a smooth penalty: square of the violation
        viol_left = np.maximum(0, -tol - left)
        viol_right = np.maximum(0, right - (1.0 + tol))
        viol_bottom = np.maximum(0, -tol - bottom)
        viol_top = np.maximum(0, top - (1.0 + tol))
        
        bound_penalty = np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)
        penalty += 100.0 * bound_penalty
        
        # 2. Overlap constraints
        # For each pair i, j
        # Overlap in x if |cx_i - cx_j| < (s_i + s_j)/2
        # Overlap in y if |cy_i - cy_j| < (s_i + s_j)/2
        # If both, they overlap.
        # Penalty is the area of intersection.
        for i in range(n):
            for j in range(i + 1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                sx = (s[i] + s[j]) / 2.0
                sy = (s[i] + s[j]) / 2.0
                
                # Overlap extent in x
                overlap_x = np.maximum(0, sx - dx)
                # Overlap extent in y
                overlap_y = np.maximum(0, sy - dy)
                
                # If both are positive, they overlap.
                # The area of the intersection is overlap_x * overlap_y
                area = overlap_x * overlap_y
                penalty += 1000.0 * area
                
        # We want to maximize sum(s), so minimize -sum(s)
        return -np.sum(s) + penalty

    # Bounds for DE
    # cx, cy in [0, 1]
    # s in [0.01, 1.0] (sides must be > 0)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.01, 1.0)] * n
    
    # Start time
    start_time = time.time()
    time_limit = 600  # seconds
    safety_margin = 10
    
    best_result = None
    best_score = -np.inf
    
    # Run Differential Evolution
    # We can run multiple times with different seeds if time permits
    seeds = [1, 2, 3, 4, 5]
    
    for seed in seeds:
        if time.time() - start_time > time_limit - safety_margin:
            break
            
        try:
            result = differential_evolution(
                objective,
                bounds,
                seed=seed,
                maxiter=1000,
                popsize=15,
                tol=1e-6,
                disp=False,
                polish=True # Use local minimization at the end of DE
            )
            
            if result.success or result.fun < best_score:
                # Check validity
                p = result.x.reshape(n, 3)
                cx, cy, s = p[:, 0], p[:, 1], p[:, 2]
                
                # Re-evaluate penalty to ensure it's valid
                # If penalty is 0 (or very small), it's valid
                # The objective includes -sum(s) + penalty.
                # If penalty is 0, fun = -sum(s).
                # We need to check if the actual penalty part is 0.
                
                # Let's calculate the penalty part separately for verification
                penalty_val = 0.0
                left = cx - s / 2.0
                right = cx + s / 2.0
                bottom = cy - s / 2.0
                top = cy + s / 2.0
                viol_left = np.maximum(0, -tol - left)
                viol_right = np.maximum(0, right - (1.0 + tol))
                viol_bottom = np.maximum(0, -tol - bottom)
                viol_top = np.maximum(0, top - (1.0 + tol))
                bound_penalty = np.sum(viol_left**2 + viol_right**2 + viol_bottom**2 + viol_top**2)
                penalty_val += 100.0 * bound_penalty
                
                for i in range(n):
                    for j in range(i + 1, n):
                        dx = abs(cx[i] - cx[j])
                        dy = abs(cy[i] - cy[j])
                        sx = (s[i] + s[j]) / 2.0
                        sy = (s[i] + s[j]) / 2.0
                        overlap_x = np.maximum(0, sx - dx)
                        overlap_y = np.maximum(0, sy - dy)
                        area = overlap_x * overlap_y
                        penalty_val += 1000.0 * area
                
                if penalty_val < 1e-9:
                    # Valid solution
                    score = np.sum(s)
                    if score > best_score:
                        best_score = score
                        best_result = (cx.copy(), cy.copy(), s.copy())
                        
        except Exception as e:
            pass

    if best_result is None:
        # Fallback to grid if DE failed
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    cx, cy, s = best_result
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    return centers, sides
