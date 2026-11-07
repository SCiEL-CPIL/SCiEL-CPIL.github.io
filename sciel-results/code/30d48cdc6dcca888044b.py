import numpy as np
import time
import scipy.optimize as opt

N = 50

def compute_value(h):
    # h is a 1D array of length N
    g = np.convolve(h, h)
    m = len(g)
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    
    # Points np.linspace(-0.5, 0.5, m + 2)
    # Spacing w
    w = 1.0 / (m + 1)
    
    # l2 = sum over i in 0..m of (w / 3) * (y[i]^2 + y[i] * y[i+1] + y[i+1]^2)
    # i goes from 0 to m inclusive
    # y indices involved: 0 to m+1
    # y has length m+2, so indices 0..m+1 are valid
    terms = (y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2)
    l2 = np.sum(terms) * (w / 3.0)
    
    # norm_1 = sum(abs(g)) / (m + 1)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    
    # norm_inf = max(abs(g))
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
        
    value = l2 / (norm_1 * norm_inf)
    return value

def objective(h):
    # We want to maximize value, so minimize -value
    v = compute_value(h)
    if not np.isfinite(v):
        return 1e10
    return -v

def construct():
    start_time = time.time()
    best_val = 0.0
    best_h = None
    
    # Seed solution value
    seed_h = np.ones(N)
    seed_val = compute_value(seed_h)
    best_val = seed_val
    best_h = seed_h
    
    # Random restarts
    # Use a fixed number of restarts, but check time
    max_restarts = 100
    rng = np.random.default_rng(42)
    
    for i in range(max_restarts):
        # Check time
        if time.time() - start_time > 110.0:
            break
            
        # Generate random initial guess
        # Scale doesn't matter, but let's keep it in a reasonable range
        # Uniform [0, 1] is a good start
        h0 = rng.random(N)
        
        # Ensure it's not zero
        if np.sum(h0) == 0:
            h0 = np.ones(N)
            
        # Optimization
        try:
            res = opt.minimize(objective, h0, method='L-BFGS-B', 
                                bounds=[(0, None)] * N,
                                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            h_opt = res.x
            val_opt = -res.fun
            
            if val_opt > best_val:
                best_val = val_opt
                best_h = h_opt
        except Exception:
            pass
            
    return list(best_h)
