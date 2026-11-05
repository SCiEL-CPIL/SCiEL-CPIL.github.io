import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 1. Initialization: Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=float)
    
    # 2. Precompute powers of z for grid
    N = 4096
    thetas = 2 * np.pi * np.arange(N) / N
    k = np.arange(24)
    Z = np.exp(1j * np.outer(thetas, k))
    
    # Refinement parameters
    REFINE_PEAKS = 5
    REFINE_POINTS = 50
    REFINE_WIDTH = 4 * (2 * np.pi / N)
    
    def true_eval(coeffs):
        P_grid = Z @ coeffs
        mags = np.abs(P_grid)
        max_grid = np.max(mags)
        
        top_indices = np.argpartition(mags, -REFINE_PEAKS)[-REFINE_PEAKS:]
        
        max_ref = 0.0
        for idx in top_indices:
            theta_c = thetas[idx]
            offsets = np.linspace(-REFINE_WIDTH/2, REFINE_WIDTH/2, REFINE_POINTS)
            thetas_ref = theta_c + offsets
            powers_ref = np.exp(1j * np.outer(thetas_ref, k))
            P_ref = powers_ref @ coeffs
            mags_ref = np.abs(P_ref)
            local_max = np.max(mags_ref)
            if local_max > max_ref:
                max_ref = local_max
                
        return max(max_grid, max_ref)

    current_c = c.copy()
    current_v = true_eval(current_c)
    best_c = current_c.copy()
    best_v = current_v
    
    T = 0.05
    cool_rate = 0.999
    min_T = 1e-6
    
    while time.time() < deadline:
        i = random.randint(0, 23)
        new_c = current_c.copy()
        new_c[i] *= -1
        
        new_v = true_eval(new_c)
        
        delta = new_v - current_v
        
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_c = new_c
            current_v = new_v
            
            if current_v < best_v:
                best_v = current_v
                best_c = current_c.copy()
                
        T = max(min_T, T * cool_rate)
        
    return [int(x) for x in best_c]
