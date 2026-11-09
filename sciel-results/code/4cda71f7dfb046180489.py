import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    """
    Pack 10 squares in a unit square to maximize sum of side lengths.
    Uses a multi-start local optimization with a soft penalty for overlaps and boundary violations.
    Falls back to a known valid packing (2 large + 8 small grid) if optimization fails or exceeds time.
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 600  # Safety margin before 630s limit
    
    # Known valid packing: 2 large (0.5) + 8 small (0.25)
    # Sum = 3.0
    def get_baseline_packing():
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        # 2 large squares in bottom half
        sides[0] = 0.5
        centers[0] = [0.25, 0.25]
        sides[1] = 0.5
        centers[1] = [0.75, 0.25]
        
        # 8 small squares in top half (4 cols, 2 rows)
        s_small = 0.25
        for i in range(8):
            row = i // 4
            col = i % 4
            sides[2 + i] = s_small
            cx = (col + 0.5) * s_small
            cy = 0.5 + (row + 0.5) * s_small
            centers[2 + i] = [cx, cy]
            
        return centers, sides

    best_centers, best_sides = get_baseline_packing()
    best_score = np.sum(best_sides)
    
    # Optimization function
    def objective(params):
        # Unpack
        cx = params[0:n]
        cy = params[n:2*n]
        s = params[2*n:3*n]
        
        # Ensure positive sides
        s = np.maximum(s, 1e-6)
        
        # Boundary penalty
        # cx should be in [s/2, 1 - s/2]
        # cy should be in [s/2, 1 - s/2]
        pen_boundary = 0.0
        for i in range(n):
            pen_boundary += np.maximum(0, s[i]/2 - cx[i])
            pen_boundary += np.maximum(0, cx[i] - (1 - s[i]/2))
            pen_boundary += np.maximum(0, s[i]/2 - cy[i])
            pen_boundary += np.maximum(0, cy[i] - (1 - s[i]/2))
            
        # Overlap penalty
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                # Overlap in x
                ox = np.maximum(0, (s[i] + s[j])/2 - np.abs(cx[i] - cx[j]))
                # Overlap in y
                oy = np.maximum(0, (s[i] + s[j])/2 - np.abs(cy[i] - cy[j]))
                # Area of overlap
                pen_overlap += ox * oy
                
        # Objective: Minimize -sum(s) + penalties
        # We want to maximize sum(s), so minimize -sum(s)
        # Penalty weights need to be large enough to enforce constraints
        weight_boundary = 100.0
        weight_overlap = 100.0
        
        return -np.sum(s) + weight_boundary * pen_boundary + weight_overlap * pen_overlap

    def is_valid(centers, sides):
        tol = 1e-7
        # Check finite
        if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
            return False
        # Check sides > 0
        if np.any(sides <= 0):
            return False
        # Check bounds
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    # Multi-start optimization
    rng = np.random.default_rng(42)
    n_restarts = 10
    attempts = 0
    max_attempts = 20
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        # Generate initial guess
        if attempts == 1:
            # Start with baseline
            c0, s0 = get_baseline_packing()
            x0 = np.concatenate([c0[:,0], c0[:,1], s0])
        else:
            # Random perturbation of baseline or random start
            if rng.random() < 0.5:
                c0, s0 = get_baseline_packing()
                # Perturb
                c0 += rng.normal(0, 0.05, c0.shape)
                s0 += rng.normal(0, 0.05, s0.shape)
                # Clip to valid range roughly
                s0 = np.clip(s0, 0.01, 0.9)
                c0[:,0] = np.clip(c0[:,0], 0, 1)
                c0[:,1] = np.clip(c0[:,1], 0, 1)
            else:
                # Random grid-like start
                c0 = rng.uniform(0.1, 0.9, (n, 2))
                s0 = rng.uniform(0.1, 0.4, n)
            x0 = np.concatenate([c0[:,0], c0[:,1], s0])
        
        # Bounds for L-BFGS-B
        # cx, cy in [0, 1]
        # s in [0.01, 1.0]
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0.01, 1.0)] * n
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
        except Exception:
            continue
            
        if res.success:
            x_opt = res.x
            cx_opt = x_opt[0:n]
            cy_opt = x_opt[n:2*n]
            s_opt = x_opt[2*n:3*n]
            
            # Check validity
            centers_opt = np.stack([cx_opt, cy_opt], axis=1)
            if is_valid(centers_opt, s_opt):
                score_opt = np.sum(s_opt)
                if score_opt > best_score:
                    best_score = score_opt
                    best_centers = centers_opt
                    best_sides = s_opt
    
    # Final validation
    if not is_valid(best_centers, best_sides):
        # Fallback to baseline
        best_centers, best_sides = get_baseline_packing()
        
    return best_centers, best_sides
