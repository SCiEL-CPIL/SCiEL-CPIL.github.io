import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    time_limit = 600.0
    
    # Baseline: 3x3 grid + 1 small square in corner
    # s = 1/3 - eps
    eps = 1e-6
    s_base = 1.0/3.0 - eps
    t_base = 1.0 - 3.0 * s_base
    
    # Centers for 3x3 grid in bottom-left
    cols = [0.5*s_base, 1.5*s_base, 2.5*s_base]
    rows = [0.5*s_base, 1.5*s_base, 2.5*s_base]
    centers_base = []
    sides_base = []
    for r in rows:
        for c in cols:
            centers_base.append([c, r])
            sides_base.append(s_base)
    # 10th square in top-right
    centers_base.append([1.0 - t_base/2.0, 1.0 - t_base/2.0])
    sides_base.append(t_base)
    
    best_centers = np.array(centers_base, dtype=np.float64)
    best_sides = np.array(sides_base, dtype=np.float64)
    best_score = np.sum(best_sides)
    
    def calculate_penalty(x):
        # x: [cx1, cy1, s1, ..., cx10, cy10, s10]
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Boundary penalties
        pen = 0.0
        # s > 0
        pen += np.sum(np.maximum(0, -s)**2)
        
        # Inside unit square
        # cx - s/2 >= 0  => cx >= s/2
        # cx + s/2 <= 1  => cx <= 1 - s/2
        pen += np.sum(np.maximum(0, s/2 - cx)**2)
        pen += np.sum(np.maximum(0, cx - (1 - s/2))**2)
        pen += np.sum(np.maximum(0, s/2 - cy)**2)
        pen += np.sum(np.maximum(0, cy - (1 - s/2))**2)
        
        # Overlap penalties
        # Vectorized calculation for pairs
        # i < j
        total_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                dist_req = (s[i] + s[j]) / 2.0
                
                # Soft penalty for violating separation
                # If dx < dist_req and dy < dist_req, they overlap
                # We penalize the "amount" of overlap in both axes
                # This encourages pushing apart
                pen_x = np.maximum(0, dist_req - dx)
                pen_y = np.maximum(0, dist_req - dy)
                # If both are violated, it's a hard overlap.
                # Simple sum of squares works well for local search
                total_pen += (pen_x**2 + pen_y**2)
        
        return pen + total_pen

    def objective(x):
        s = x[2::3]
        return -np.sum(s) + calculate_penalty(x)
        
    def is_valid(centers, sides, tol=1e-7):
        # Check boundaries
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if not (s > 0):
                return False
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                cx_i, cy_i = centers[i]
                cx_j, cy_j = centers[j]
                s_i = sides[i]
                s_j = sides[j]
                
                dx = abs(cx_i - cx_j)
                dy = abs(cy_i - cy_j)
                t = (s_i + s_j) / 2.0 - tol
                
                if dx < t and dy < t:
                    return False
        return True

    def run_optimization(x0):
        # Use Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
            return res.x
        except Exception:
            return x0

    # Restart loop
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
            
        # Generate initial guess
        if r == 0:
            # Use baseline
            x0 = np.zeros(n*3)
            x0[0::3] = best_centers[:, 0]
            x0[1::3] = best_centers[:, 1]
            x0[2::3] = best_sides
        else:
            # Perturb baseline
            x0 = np.zeros(n*3)
            x0[0::3] = best_centers[:, 0] + np.random.normal(0, 0.05, n)
            x0[1::3] = best_centers[:, 1] + np.random.normal(0, 0.05, n)
            x0[2::3] = best_sides + np.random.normal(0, 0.02, n)
            x0[2::3] = np.maximum(x0[2::3], 0.01)
            
            # Or random
            if r % 5 == 0:
                x0[0::3] = np.random.uniform(0.1, 0.9, n)
                x0[1::3] = np.random.uniform(0.1, 0.9, n)
                x0[2::3] = np.random.uniform(0.1, 0.4, n)

        x_opt = run_optimization(x0)
        
        cx = x_opt[0::3]
        cy = x_opt[1::3]
        s = x_opt[2::3]
        
        centers = np.column_stack((cx, cy))
        sides = s
        
        if is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers
                best_sides = sides

    return best_centers, best_sides
