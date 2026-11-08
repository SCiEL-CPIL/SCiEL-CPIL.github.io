import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sequence of 50 nonnegative heights.
    Uses grid search and local optimization to find a configuration
    that maximizes ||h||_2^2 / (||h||_1 * ||h*h||_inf).
    """
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    def compute_ratio(h):
        h_arr = np.array(h, dtype=float)
        l1 = np.sum(h_arr)
        if l1 < 1e-15:
            return 0.0
        l2_sq = np.sum(h_arr**2)
        hh = np.convolve(h_arr, h_arr, mode='full')
        sup_norm = np.max(hh)
        if sup_norm < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_norm)
    
    # Initialize with known good 2-point cluster
    best_h = [0.0] * N
    best_h[0] = 2.0 / 3.0
    best_h[1] = 1.0 / 3.0
    best_ratio = compute_ratio(best_h)
    
    # Grid search over 2-point configurations with different positions
    for pos in range(N - 1):
        for ratio_val in np.linspace(0.5, 2.0, 16):
            a = ratio_val / (1.0 + ratio_val)
            b = 1.0 / (1.0 + ratio_val)
            h = [0.0] * N
            h[pos] = a
            h[pos + 1] = b
            r = compute_ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h[:]
    
    if time.time() - start_time > deadline:
        return best_h
    
    # Grid search over 3-point configurations
    for pos in range(N - 2):
        for a in np.linspace(0.2, 0.8, 7):
            for b in np.linspace(0.05, 1.0 - a - 0.05, 6):
                c = 1.0 - a - b
                if c < 0.01:
                    continue
                h = [0.0] * N
                h[pos] = a
                h[pos + 1] = b
                h[pos + 2] = c
                r = compute_ratio(h)
                if r > best_ratio:
                    best_ratio = r
                    best_h = h[:]
    
    if time.time() - start_time > deadline:
        return best_h
    
    # Try local optimization using Nelder-Mead
    try:
        from scipy.optimize import minimize
        
        def neg_ratio(params):
            h = [0.0] * N
            for i in range(min(len(params), N)):
                h[i] = max(0.0, float(params[i]))
            return -compute_ratio(h)
        
        # Optimize from best found configuration
        x0 = np.array(best_h[:6], dtype=float)
        result = minimize(neg_ratio, x0, method='Nelder-Mead',
                         options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8,
                                  'adaptive': True})
        
        if result.fun < -best_ratio:
            h = [0.0] * N
            for i in range(6):
                h[i] = max(0.0, float(result.x[i]))
            r = compute_ratio(h)
            if r > best_ratio:
                best_ratio = r
                best_h = h[:]
    except Exception:
        pass
    
    return best_h
# EVOLVE-BLOCK-END
