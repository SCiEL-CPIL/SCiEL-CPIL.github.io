import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)  # spacing for linspace(-0.5, 0.5, m+2)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + y[:m+1]*y[1:] + y[1:]**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _negative_value(h):
    return -_compute_value(h)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Start from uniform seed
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    # Use multiple random restarts and local optimization
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            h_init = h0.copy()
        else:
            # Random initialization: mix of uniform and random
            h_init = np.random.exponential(1.0, size=N)
            h_init = h_init / np.max(h_init) * np.random.uniform(0.5, 1.5)
        
        # Optimize
        res = minimize(
            _negative_value,
            h_init,
            method='L-BFGS-B',
            bounds=[(0, None)] * N,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_val * -1:
            h_opt = np.maximum(res.x, 0)
            val = _compute_value(h_opt)
            if val > best_val:
                best_val = val
                best_h = h_opt.copy()
    
    return list(best_h)
