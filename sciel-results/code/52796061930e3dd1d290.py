import numpy as np
from scipy.optimize import minimize
import time
import warnings

N = 50

# EVOLVE-BLOCK-START
def construct():
    def score(h):
        h = np.array(h)
        if np.any(h < 0) or np.sum(h) == 0:
            return 0.0
        # Normalize to sum 1 to remove scale invariance
        h_norm = h / np.sum(h)
        l2_sq = np.sum(h_norm ** 2)
        conv = np.convolve(h_norm, h_norm, mode='full')
        max_conv = np.max(conv)
        if max_conv == 0:
            return 0.0
        return l2_sq / max_conv

    # Initial guesses
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Candidate 1: [2/3, 1/3, 0...]
    h1 = np.zeros(N)
    h1[0] = 2/3
    h1[1] = 1/3
    s1 = score(h1)
    if s1 > best_score:
        best_score = s1
        best_h = h1.copy()
        
    # Candidate 2: [2/3, 1/4, 1/4, 0...] (from analysis c=1/4)
    h2 = np.zeros(N)
    h2[0] = 2/3
    h2[1] = 1/3
    h2[2] = 1/4
    # Normalize
    h2 = h2 / np.sum(h2)
    s2 = score(h2)
    if s2 > best_score:
        best_score = s2
        best_h = h2.copy()

    # Run local optimization on a subset of dimensions (first 10) to find better local max
    # We fix the rest to 0 to keep it sparse
    def objective(x):
        # x is the first 10 elements
        h_full = np.zeros(N)
        h_full[:len(x)] = x
        # Ensure non-negative
        h_full = np.abs(h_full)
        if np.sum(h_full) == 0:
            return 0.0
        return -score(h_full)
    
    # Start from best_h truncated to 10
    x0 = best_h[:10].copy()
    
    # Use Nelder-Mead
    start_time = time.time()
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and time.time() - start_time < 10:
            h_opt = np.zeros(N)
            h_opt[:len(res.x)] = np.abs(res.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
    except Exception:
        pass

    # Try a few random restarts with small support
    for _ in range(5):
        if time.time() - start_time > 10:
            break
        # Random sparse vector
        support_size = np.random.randint(2, 8)
        indices = np.random.choice(N, support_size, replace=False)
        indices.sort()
        h_rand = np.zeros(N)
        h_rand[indices] = np.random.rand(support_size)
        h_rand = h_rand / np.sum(h_rand)
        
        x0_rand = h_rand[:10].copy()
        try:
            res = minimize(objective, x0_rand, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            h_opt = np.zeros(N)
            h_opt[:len(res.x)] = np.abs(res.x)
            s_opt = score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Final check and return
    # Ensure non-negative
    best_h = np.abs(best_h)
    if np.sum(best_h) == 0:
        best_h[0] = 1.0
    else:
        best_h = best_h / np.sum(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
