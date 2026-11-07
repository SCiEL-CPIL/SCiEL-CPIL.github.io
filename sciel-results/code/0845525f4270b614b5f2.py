import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # seconds, safety margin before 150s

def compute_value(h):
    h = np.asarray(h, dtype=np.float64)
    if np.any(h < -1e-15) or not np.all(np.isfinite(h)):
        return -np.inf
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)
    w = 1.0 / (m + 1)
    y = np.concatenate(([0.0], g, [0.0]))
    # l2 = sum_{i=0}^{m} (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    y2 = y**2
    cross = y * np.roll(y, -1)
    # y[i]*y[i+1] for i=0..m: that's y[0]*y[1], ..., y[m]*y[m+1]
    # y has length m+2, indices 0..m+1
    # We need i from 0 to m, so y[i] and y[i+1] for i=0..m
    cross_terms = y[:m+1] * y[1:m+2]
    l2 = (w / 3.0) * np.sum(y[:m+1]**2 + cross_terms + y[1:m+2]**2)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf < 1e-30:
        return -np.inf
    return l2 / (norm_1 * norm_inf)

def negative_value(h):
    v = compute_value(h)
    if np.isinf(v):
        return 1e10
    return -v

def construct():
    start_time = time.time()
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    # Multi-start optimization
    np.random.seed(42)
    n_starts = 30
    for i in range(n_starts):
        if time_left() < 3.0:
            break
        # Generate random starting point
        if i == 0:
            h0 = np.ones(N)
        elif i == 1:
            h0 = np.ones(N) * 0.5
        elif i == 2:
            # Triangle
            h0 = np.linspace(1, 0.1, N)
        elif i == 3:
            # Reverse triangle
            h0 = np.linspace(0.1, 1, N)
        elif i == 4:
            # Peak in middle
            h0 = np.exp(-0.5 * ((np.arange(N) - N/2) / (N/4))**2)
        elif i == 5:
            # Two peaks
            h0 = np.exp(-0.5 * ((np.arange(N) - N*0.25) / (N/6))**2) + np.exp(-0.5 * ((np.arange(N) - N*0.75) / (N/6))**2)
        else:
            h0 = np.random.rand(N) * 1.5
        
        # Scale to have max(g) ~ 1
        g0 = np.convolve(h0, h0)
        if np.max(g0) > 0:
            h0 = h0 / np.sqrt(np.max(g0))
        
        res = minimize(negative_value, h0, method='L-BFGS-B', 
                      bounds=[(0, None)]*N,
                      options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time_left() < 3.0:
            break
        
        h_opt = np.maximum(res.x, 0)
        val = compute_value(h_opt)
        if val > best_val:
            best_val = val
            best_h = h_opt
    
    return list(best_h)
