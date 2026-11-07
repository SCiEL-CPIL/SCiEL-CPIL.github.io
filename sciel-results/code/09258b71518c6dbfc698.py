import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _value(h):
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,):
        return np.inf
    if np.any(~np.isfinite(h)):
        return np.inf
    if np.any(h < -1e-12) or np.any(h > 1 + 1e-12):
        return np.inf
    if abs(np.sum(h) - N / 2.0) > 1e-9 * N:
        return np.inf
    corr = np.correlate(h, 1.0 - h, mode="full")
    return float(np.max(corr)) / N * 2.0

def construct():
    t_start = time.time()
    deadline = t_start + 140.0
    
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    bounds = [(0.0, 1.0)] * N
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0}]
    
    def objective(x):
        h = np.clip(x, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode="full")
        return np.max(corr)
        
    def try_optimize(start_h, maxiter=100):
        nonlocal best_h, best_val
        try:
            res = minimize(
                objective, start_h,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': maxiter, 'ftol': 1e-12}
            )
            h_candidate = np.clip(res.x, 0.0, 1.0)
            # Fix sum drift
            diff = N/2.0 - np.sum(h_candidate)
            if abs(diff) > 1e-9:
                h_candidate = h_candidate + diff / N
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
                diff = N/2.0 - np.sum(h_candidate)
                if abs(diff) > 1e-9:
                     h_candidate = h_candidate + diff / N
                     h_candidate = np.clip(h_candidate, 0.0, 1.0)
            
            val = _value(h_candidate)
            if val < best_val:
                best_h = h_candidate
                best_val = val
        except Exception:
            pass

    # 1. Try seed
    try_optimize(h0, maxiter=200)
    
    if time.time() > deadline:
        return best_h.tolist()

    # 2. Random restarts
    rng = np.random.default_rng(42)
    num_restarts = 30
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        h_rand = rng.uniform(0, 1, N)
        diff = 50.0 - np.sum(h_rand)
        h_rand += diff / N
        h_rand = np.clip(h_rand, 0.0, 1.0)
        
        try_optimize(h_rand)
        
        # Also try some structured starts
        if i % 3 == 0:
             # Sine wave start
             freq = rng.uniform(1, 10)
             phase = rng.uniform(0, 2*np.pi)
             h_sine = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N + phase)
             h_sine = np.clip(h_sine, 0, 1)
             diff = 50.0 - np.sum(h_sine)
             h_sine += diff / N
             h_sine = np.clip(h_sine, 0, 1)
             try_optimize(h_sine)

    return best_h.tolist()
