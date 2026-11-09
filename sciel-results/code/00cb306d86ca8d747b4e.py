import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0
    
    # Initial guess: Grid
    k = 3
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    x0 = np.zeros(3 * n)
    x0[0::3] = cx
    x0[1::3] = cy
    x0[2::3] = side
    
    best_x = x0.copy()
    best_score = np.sum(side)
    
    def unpack(x):
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        return cx, cy, s

    def penalty_and_objective(x):
        cx, cy, s = unpack(x)
        
        # Ensure s > 0 (clamp to small positive)
        s_clamped = np.maximum(s, 1e-6)
        
        # Boundary penalty
        # x range: [s/2, 1-s/2]
        # Penalty if cx < s/2 or cx > 1-s/2
        px_low = np.maximum(0, s_clamped/2 - cx)
        px_high = np.maximum(0, cx - (1 - s_clamped/2))
        py_low = np.maximum(0, s_clamped/2 - cy)
        py_high = np.maximum(0, cy - (1 - s_clamped/2))
        bound_pen = np.sum(px_low**2) + np.sum(px_high**2) + np.sum(py_low**2) + np.sum(py_high**2)
        
        # Overlap penalty
        # For each pair i, j
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                req_x = (s_clamped[i] + s_clamped[j]) / 2.0
                req_y = (s_clamped[i] + s_clamped[j]) / 2.0
                
                ox = req_x - dx
                oy = req_y - dy
                
                if ox > 0 and oy > 0:
                    # Overlap area
                    overlap_pen += ox * oy
        
        # Total penalty
        total_pen = bound_pen + overlap_pen
        
        # Objective: Maximize sum of sides -> Minimize -sum(sides) + lambda * penalty
        # Lambda needs to be large
        lam = 100.0
        obj = -np.sum(s_clamped) + lam * total_pen
        
        return obj

    # Run local optimization with multiple restarts
    # We will use Nelder-Mead
    # To improve results, we can also try perturbing the initial grid
    
    rng = np.random.default_rng(42)
    
    # Define a list of initial guesses
    initial_guesses = [x0]
    
    # Generate a few random valid packings (greedy)
    # This is a bit expensive, so maybe just perturb the grid
    for _ in range(5):
        x_pert = x0.copy()
        x_pert += rng.normal(0, 0.05, size=x_pert.shape)
        # Clamp sides to be positive
        x_pert[2::3] = np.maximum(x_pert[2::3], 0.01)
        initial_guesses.append(x_pert)
        
    # Also try a "2+8" style guess
    # 2 large squares, 8 small
    x_28 = np.zeros(3*n)
    # Square 0: top-left, large
    x_28[0] = 0.3
    x_28[1] = 0.7
    x_28[2] = 0.6
    # Square 1: bottom-right, large
    x_28[3] = 0.7
    x_28[4] = 0.3
    x_28[5] = 0.6
    # Others: small, scattered
    for i in range(2, 10):
        x_28[i*3] = rng.uniform(0.1, 0.9)
        x_28[i*3+1] = rng.uniform(0.1, 0.9)
        x_28[i*3+2] = 0.2
    initial_guesses.append(x_28)
    
    # Optimize each guess
    for x_init in initial_guesses:
        if time.time() > deadline - safety_margin:
            break
            
        try:
            res = minimize(
                penalty_and_objective, 
                x_init, 
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-8,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            
            # Check if result is valid (low penalty)
            cx, cy, s = unpack(res.x)
            s_clamped = np.maximum(s, 1e-6)
            
            # Verify validity manually to be safe
            valid = True
            for i in range(n):
                if s_clamped[i] <= 0: valid = False; break
                if cx[i] - s_clamped[i]/2 < -1e-7 or cx[i] + s_clamped[i]/2 > 1 + 1e-7: valid = False; break
                if cy[i] - s_clamped[i]/2 < -1e-7 or cy[i] + s_clamped[i]/2 > 1 + 1e-7: valid = False; break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = abs(cx[i] - cx[j])
                        dy = abs(cy[i] - cy[j])
                        t = (s_clamped[i] + s_clamped[j])/2 - 1e-7
                        if dx < t and dy < t:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                score = np.sum(s_clamped)
                if score > best_score:
                    best_score = score
                    best_x = res.x.copy()
                    
        except Exception as e:
            pass
            
    # Final extraction
    cx, cy, s = unpack(best_x)
    # Ensure s is strictly positive
    s = np.maximum(s, 1e-9)
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides
