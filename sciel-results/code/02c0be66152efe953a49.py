import numpy as np
from scipy.optimize import minimize
import time

N = 100
DEADLINE = 145.0  # 5s margin

def objective(h, M=10.0):
    corr = np.correlate(h, 1-h, mode="full")
    max_corr = M * np.log(np.sum(np.exp(corr/M)))
    sum_penalty = (np.sum(h) - N/2)**2
    return max_corr / N * 2 + 1000 * sum_penalty

def construct():
    start_time = time.time()
    bounds = [(0, 1) for _ in range(N)]
    
    # Try a few initializations
    best_h = None
    best_val = float('inf')
    
    initializations = [
        np.full(N, 0.5),
        np.concatenate([np.ones(N//2), np.zeros(N//2)]),
        np.random.rand(N)
    ]
    
    for h0 in initializations:
        if time.time() > DEADLINE:
            break
            
        h0 = h0.copy()
        h0 = h0 + (N/2 - np.sum(h0))/N
        h0 = np.clip(h0, 0, 1)
        
        try:
            res = minimize(objective, h0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12})
            h = res.x
            h = h + (N/2 - np.sum(h))/N
            h = np.clip(h, 0, 1)
            
            # Calculate true value
            corr = np.correlate(h, 1-h, mode="full")
            val = np.max(corr) / N * 2
            if val < best_val:
                best_val = val
                best_h = h
        except Exception:
            pass
            
    if best_h is None:
        return np.full(N, 0.5)
        
    return best_h
