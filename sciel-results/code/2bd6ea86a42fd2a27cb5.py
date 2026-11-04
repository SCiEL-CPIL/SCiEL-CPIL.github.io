import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 115.0  # Safety margin before 120s limit

def _score(h):
    """Compute the score: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    max_val = np.max(conv)
    if max_val <= 0:
        return 0.0
    return l2_sq / max_val

def _score_neg(h):
    return -_score(h)

def construct():
    start_time = time.time()
    
    # Initialize with best known heuristics
    # 1. Uniform
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # 2. Sidon-like sets
    # A Sidon set (or B2 set) is a set of integers where all pairwise sums are distinct.
    # For small N, we can hardcode or generate simple ones.
    # Example Sidon set in [0, 49]: {0, 1, 4, 10, 18, 23, 31, 37, 46}
    sidon_sets = [
        [0, 1, 4, 10, 18, 23, 31, 37, 46],
        [0, 2, 5, 11, 17, 24, 30, 38, 45],
        [0, 3, 6, 12, 19, 25, 33, 39, 48],
        [0, 1, 3, 7, 12, 18, 25, 31, 40],
    ]
    
    for idx_set in sidon_sets:
        h = np.zeros(N)
        h[idx_set] = 1.0
        s = _score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
            
    # 3. Random sparse initializations
    rng = np.random.RandomState(42)
    candidates = [best_h.copy()]
    
    for i in range(50):
        k = rng.randint(5, 15)
        idx = rng.choice(N, k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        candidates.append(h)
        
    # Sort candidates by score
    candidates.sort(key=_score, reverse=True)
    
    # Optimization loop
    # We will use L-BFGS-B to maximize the score directly.
    # The objective is non-smooth due to the max, but L-BFGS-B often works well enough.
    # To improve robustness, we can also try a smooth approximation if time permits, 
    # but given the deadline, direct optimization from good starts is safer.
    
    for i in range(min(10, len(candidates))):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        h0 = candidates[i]
        
        # Direct optimization
        try:
            res = minimize(_score_neg, h0, method='L-BFGS-B', 
                           bounds=[(0, None)] * N,
                           options={'maxiter': 50, 'gtol': 1e-8, 'maxls': 20})
            h_opt = np.maximum(res.x, 0.0)
            s_opt = _score(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
        # If we still have time, do a second pass with more iterations on the best found so far
        if i == 0 and time.time() - start_time < DEADLINE - 20:
            try:
                res2 = minimize(_score_neg, best_h, method='L-BFGS-B', 
                                bounds=[(0, None)] * N,
                                options={'maxiter': 200, 'gtol': 1e-10, 'maxls': 20})
                h_opt2 = np.maximum(res2.x, 0.0)
                s_opt2 = _score(h_opt2)
                if s_opt2 > best_score:
                    best_score = s_opt2
                    best_h = h_opt2.copy()
            except Exception:
                pass

    return best_h.tolist()
