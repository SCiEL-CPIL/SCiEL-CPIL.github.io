import numpy as np
import scipy.optimize as so
import time

N = 50

def compute_value(h):
    h = np.asarray(h)
    if not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:m+1] = g
    w = 0.01
    # Vectorized sum
    # y[i]^2 + y[i]y[i+1] + y[i+1]^2
    # i from 0 to m
    # y[0..m] and y[1..m+1]
    term = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = np.sum(term) * w / 3.0
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf < 1e-12:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def objective(x):
    h = x**2
    val = compute_value(h)
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    best_val = 0.0
    best_h = [1.0] * N
    
    # Initializations
    # 1. Seed
    x0_seed = np.ones(N)
    # 2. Random
    rng = np.random.default_rng(42)
    inits = [x0_seed]
    for _ in range(5):
        inits.append(rng.random(N) * 2.0)
        
    for x0 in inits:
        if time.time() > deadline:
            break
        
        try:
            # L-BFGS-B is fast and handles bounds
            # We don't strictly need bounds on x, but let's keep them reasonable
            res = so.minimize(objective, x0, method='L-BFGS-B', 
                              options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            
            if res.fun < -best_val: # res.fun is -value
                best_val = -res.fun
                best_h = res.x**2
        except Exception:
            pass

    # Final check
    if best_val <= 0:
        return [1.0] * N
        
    return list(best_h)
