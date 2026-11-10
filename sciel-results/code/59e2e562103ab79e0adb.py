# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    
    def stable_softplus(x, k=1.0):
        # (1/k) * log(1 + exp(k*x))
        # = max(0, x) + (1/k) * log(1 + exp(-k*|x|))
        return np.maximum(0, x) + np.log1p(np.exp(-k * np.abs(x))) / k

    def objective(vars):
        x = vars[0::3]
        y = vars[1::3]
        s = vars[2::3]
        
        score = -np.sum(s)
        
        k = 10.0
        M_bound = 1000.0
        M_overlap = 1000.0
        
        # Boundary penalty
        pen_bound = 0.0
        for i in range(n):
            dl = s[i]/2.0 - x[i]
            dr = x[i] + s[i]/2.0 - 1.0
            dt = s[i]/2.0 - y[i]
            db = y[i] + s[i]/2.0 - 1.0
            # Square the softplus to make penalty quadratic-like for large violations
            pen_bound += stable_softplus(dl, k)**2 + stable_softplus(dr, k)**2 + \
                         stable_softplus(dt, k)**2 + stable_softplus(db, k)**2
            
        # Overlap penalties
        pen_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                A = (s[i] + s[j]) / 2.0 - np.abs(x[i] - x[j])
                B = (s[i] + s[j]) / 2.0 - np.abs(y[i] - y[j])
                sp_A = stable_softplus(A, k)
                sp_B = stable_softplus(B, k)
                pen_overlap += sp_A * sp_B
                
        return score + M_bound * pen_bound + M_overlap * pen_overlap

    bounds = [(0, 1)] * (3 * n)
    
    # Initial guess: Grid
    k_grid = 3
    side_grid = 0.9 / k_grid
    idx = np.arange(n)
    cols = idx % k_grid
    rows = idx // k_grid
    cx_init = (cols + 0.5) * (1.0/k_grid)
    cy_init = (rows + 0.5) * (1.0/k_grid)
    
    x_init = cx_init.copy()
    y_init = cy_init.copy()
    s_init = np.full(n, side_grid)
    
    vars_init = np.empty(3*n)
    vars_init[0::3] = x_init
    vars_init[1::3] = y_init
    vars_init[2::3] = s_init
    
    best_vars = vars_init
    best_score = np.sum(s_init) # Baseline score
    
    def is_valid(x, y, s):
        tol = 1e-7
        if np.any(s <= 0):
            return False
        for i in range(n):
            if x[i] - s[i]/2 < -tol or x[i] + s[i]/2 > 1 + tol:
                return False
            if y[i] - s[i]/2 < -tol or y[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j])/2 - tol
                if abs(x[i] - x[j]) < t and abs(y[i] - y[j]) < t:
                    return False
        return True

    start_time = time.time()
    time_limit = 500.0
    
    for restart in range(10):
        if time.time() - start_time > time_limit:
            break
            
        if restart == 0:
            v0 = vars_init
        else:
            v0 = vars_init + np.random.normal(0, 0.05, 3*n)
            v0 = np.clip(v0, 0, 1)
            
        # L-BFGS-B
        try:
            res = minimize(objective, v0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        except Exception:
            continue
        
        if res.success:
            v_res = res.x
            x_r = v_res[0::3]
            y_r = v_res[1::3]
            s_r = v_res[2::3]
            
            if is_valid(x_r, y_r, s_r):
                current_sum_s = np.sum(s_r)
                if current_sum_s > best_score:
                    best_score = current_sum_s
                    best_vars = v_res

    x_best = best_vars[0::3]
    y_best = best_vars[1::3]
    s_best = best_vars[2::3]
    
    centers = np.stack([x_best, y_best], axis=1)
    sides = s_best
    
    return centers, sides

# EVOLVE-BLOCK-END
