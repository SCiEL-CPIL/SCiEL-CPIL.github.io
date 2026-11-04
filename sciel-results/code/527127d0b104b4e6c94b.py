import numpy as np
from scipy.optimize import minimize
import time

N = 600

def compute_score_fine(heights):
    heights = np.asarray(heights, dtype=float)
    if np.any(heights < 0):
        return 1e10
    integral = np.sum(heights)
    if integral <= 0:
        return 1e10
    autoconv0 = np.sum(heights**2)
    return autoconv0 / (integral**2)

def construct():
    start_time = time.time()
    deadline = 140.0
    
    best_heights = np.ones(N)
    best_score = compute_score_fine(best_heights)
    
    # Coarse grid surrogate: use a smaller N
    N_coarse = 60
    def compute_score_coarse(heights_coarse):
        # Upsample to fine grid
        heights_fine = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, N_coarse), heights_coarse)
        return compute_score_fine(heights_fine)
    
    # Optimize on coarse grid
    def objective(params):
        heights_coarse = np.exp(params)
        return compute_score_coarse(heights_coarse)
    
    try:
        result = minimize(objective, np.zeros(N_coarse), method='L-BFGS-B',
                         options={'maxiter': 100, 'maxfun': 10000})
        if time.time() - start_time > deadline:
            return best_heights.tolist()
        heights_coarse_opt = np.exp(result.x)
        heights_fine_opt = np.interp(np.linspace(0, 1, N), np.linspace(0, 1, N_coarse), heights_coarse_opt)
        score = compute_score_fine(heights_fine_opt)
        if score < best_score:
            best_score = score
            best_heights = heights_fine_opt
    except Exception:
        pass
    
    # Refine on fine grid
    def objective_fine(params):
        heights = np.exp(params)
        return compute_score_fine(heights)
    
    try:
        if time.time() - start_time < deadline - 20:
            result = minimize(objective_fine, np.log(np.maximum(best_heights, 1e-10)), method='L-BFGS-B',
                             options={'maxiter': 50, 'maxfun': 5000})
            if time.time() - start_time > deadline:
                return best_heights.tolist()
            new_heights = np.exp(result.x)
            new_score = compute_score_fine(new_heights)
            if new_score < best_score:
                best_heights = new_heights
                best_score = new_score
    except Exception:
        pass
    
    return best_heights.tolist()
