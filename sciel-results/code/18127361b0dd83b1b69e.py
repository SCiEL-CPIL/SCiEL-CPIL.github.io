import numpy as np
import time

N = 400

def compute_value(h):
    S = np.sum(h)
    if abs(S) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (S * S))

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_h = np.ones(N)
    best_val = 2.0
    
    # Try geometric sequences
    for r in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01, 1.05, 1.1, 1.3, 1.5, 2.0]:
        if time.time() > deadline:
            break
        h = r ** np.arange(N)
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Try linear ramps
    for slope in [0.1, 0.5, 1.0, 2.0, 5.0, -0.1, -0.5, -1.0, -2.0]:
        if time.time() > deadline:
            break
        h = 1.0 + slope * np.arange(N) / N
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Random search
    np.random.seed(123)
    for i in range(10000):
        if time.time() > deadline:
            break
        h = np.random.randn(N)
        val = compute_value(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h
