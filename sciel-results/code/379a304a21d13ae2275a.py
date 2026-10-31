import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    indices = sorted(indices)
    sums = set()
    for i in range(len(indices)):
        for j in range(i, len(indices)):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def greedy_sidon(n, max_iter=1000):
    """Find a large Sidon set in [0, n) using randomized greedy."""
    best_set = set()
    for _ in range(max_iter):
        current = set()
        # Random start
        start = np.random.randint(n)
        current.add(start)
        
        # Try to add elements in random order
        candidates = list(range(n))
        np.random.shuffle(candidates)
        
        for idx in candidates:
            if idx in current:
                continue
            # Check if adding idx is valid
            valid = True
            for existing in current:
                s = idx + existing
                # Check if s is already in sums
                # We need to maintain the set of sums for efficiency, 
                # but for small N, recalculating or checking against current sums is fine.
                # Actually, let's just check against all pairs in current + idx
                pass # Simplified: just use is_sidon_set on list(current)+[idx]
            
            # Efficient check:
            test_list = list(current) + [idx]
            if is_sidon_set(test_list):
                current.add(idx)
                
        if len(current) > len(best_set):
            best_set = set(current)
            if len(best_set) > 10: # Heuristic: if we find a very large one, stop early?
                # No, keep searching for a bit
                pass
    return list(best_set)

def optimize_weights_for_support(indices, deadline):
    """Optimize weights for a fixed support using Nelder-Mead."""
    k = len(indices)
    if k == 0:
        return np.zeros(N), 0.0
        
    # Initial guess: uniform
    x0 = np.ones(k) / k
    
    def neg_ratio(x):
        x = np.maximum(x, 0.0)
        s = np.sum(x)
        if s < 1e-15:
            return 1e10
        x = x / s
        h = np.zeros(N)
        h[indices] = x
        return -compute_ratio(h)
    
    try:
        res = minimize(neg_ratio, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        x_opt = np.maximum(res.x, 0.0)
        s = np.sum(x_opt)
        if s < 1e-15:
            return np.zeros(N), 0.0
        x_opt = x_opt / s
        
        h = np.zeros(N)
        h[indices] = x_opt
        sc = compute_ratio(h)
        return h, sc
    except Exception:
        return np.zeros(N), 0.0

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    np.random.seed(42)
    
    # Phase 1: Find large Sidon sets
    # We want the largest possible Sidon set.
    # For N=50, max size is likely 7 or 8.
    # Run greedy_sidon many times to find the best one.
    
    best_sidon = []
    num_restarts = 200
    for i in range(num_restarts):
        if time.time() > deadline - 60.0:
            break
        s = greedy_sidon(N, max_iter=50) # Inner max_iter is small, outer loop does the work
        if len(s) > len(best_sidon):
            best_sidon = s
            # If we find a very large one, we can reduce effort
            if len(best_sidon) > 9:
                break
                
    # Also try some known good patterns if search fails
    known_patterns = [
        [0, 1, 4, 10, 12, 17, 20, 24],
        [0, 1, 4, 9, 15, 21, 27, 33],
        [0, 2, 5, 11, 14, 20, 26, 32],
        [0, 1, 3, 7, 12, 18, 25, 31],
    ]
    for p in known_patterns:
        if len(p) > len(best_sidon):
            if is_sidon_set(p):
                best_sidon = p
                
    candidates = []
    
    # Evaluate the best Sidon set with uniform weights
    if best_sidon:
        h = np.zeros(N)
        for idx in best_sidon:
            h[idx] = 1.0
        sc = compute_ratio(h)
        candidates.append((sc, h.copy(), best_sidon.copy()))
        
    # Phase 2: Random sparse supports
    # Try random sets of size k from 3 to 12
    for k in range(3, 13):
        if time.time() > deadline - 40.0:
            break
        for _ in range(50):
            if time.time() > deadline - 40.0:
                break
            indices = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[indices] = 1.0
            sc = compute_ratio(h)
            candidates.append((sc, h.copy(), indices.tolist()))
            
    # Sort candidates by score
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 3: Optimize weights for top candidates
    top_k = 10
    for i, (sc, h, indices) in enumerate(candidates[:top_k]):
        if time.time() > deadline - 10.0:
            break
        
        # Only optimize if the support size is reasonable
        if len(indices) < 3:
            continue
            
        h_opt, sc_opt = optimize_weights_for_support(indices, deadline)
        if sc_opt > best_score:
            best_score = sc_opt
            best_h = h_opt
            
    # Phase 4: Gradient polish on the best solution found so far
    if time.time() < deadline - 5.0:
        # Use L-BFGS-B with smoothing on the full vector
        h = best_h.copy()
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s > 1e-15:
            h = h / s
            
        def neg_ratio_smooth(x, beta=50.0):
            x = np.maximum(x, 0.0)
            s = np.sum(x)
            if s < 1e-15:
                return 1e10
            x = x / s
            h2 = np.sum(x**2)
            conv = fftconvolve(x, x, mode='full')
            conv_clipped = np.clip(conv, 1e-15, None)
            logsumexp_val = np.log(np.sum(np.exp(beta * conv_clipped))) / beta
            if logsumexp_val < 1e-15:
                return 1e10
            return -h2 / logsumexp_val
            
        try:
            res = minimize(neg_ratio_smooth, h, args=(50.0,),
                          method='L-BFGS-B',
                          bounds=[(0.0, None)] * N,
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            h_try = np.maximum(res.x, 0.0)
            s_try = np.sum(h_try)
            if s_try > 1e-15:
                h_try = h_try / s_try
                sc_try = compute_ratio(h_try)
                if sc_try > best_score:
                    best_score = sc_try
                    best_h = h_try
        except Exception:
            pass

    return list(best_h)
