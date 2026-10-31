"""Combined program: Sidon-set structured initialization + log-space L-BFGS-B refinement."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconv_max(h):
    """Compute supremum norm of autoconvolution."""
    c = np.correlate(h, h, mode='full')
    return np.max(c)

def _ratio(h):
    """Compute the objective ratio."""
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    max_conv = _autoconv_max(h)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

def _neg_ratio(h):
    return -_ratio(h)

def is_sidon(A):
    """Check if A is a Sidon set (all sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(A)):
        for j in range(i, len(A)):
            s = A[i] + A[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_set_greedy(n):
    """Greedy construction of a Sidon set."""
    A = []
    used_sums = set()
    for x in range(n):
        valid = True
        new_sums = []
        for a in A:
            s = a + x
            if s in used_sums:
                valid = False
                break
            new_sums.append(s)
        if 2 * x in used_sums:
            valid = False
        
        if valid:
            A.append(x)
            used_sums.update(new_sums)
            used_sums.add(2 * x)
    return A

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Start with a single spike (known good: ratio=1)
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_val = _ratio(best_h)
    
    # Try a few initial patterns
    patterns = []
    # Single spike at center
    p = np.zeros(N); p[N//2] = 1.0; patterns.append(p)
    # Two spikes
    p = np.zeros(N); p[0] = 1.0; p[N//2] = 1.0; patterns.append(p)
    # Triangular
    p = np.linspace(1, 0, N//2); p = np.concatenate([p, p[::-1][:N//2]]); patterns.append(p)
    # Sparse random
    np.random.seed(42)
    for trial in range(5):
        p = np.zeros(N)
        idx = np.random.choice(N, size=5, replace=False)
        p[idx] = 1.0
        patterns.append(p)
    
    for p in patterns:
        v = _ratio(p)
        if v > best_val:
            best_val = v
            best_h = p.copy()
    
    # Sidon set based initialization
    A_greedy = find_sidon_set_greedy(N)
    h_sidon = np.zeros(N)
    for idx in A_greedy:
        h_sidon[idx] = 1.0
    if np.sum(h_sidon) > 0:
        h_sidon /= np.sum(h_sidon)
        v = _ratio(h_sidon)
        if v > best_val:
            best_val = v
            best_h = h_sidon.copy()
    
    # Try to find a larger Sidon set with random restarts
    best_A = A_greedy
    for target_size in range(len(A_greedy) + 1, 10):
        found = False
        for _ in range(200): # 200 random tries per size
            if time.time() > deadline - 20:
                break
            A_try = sorted(np.random.choice(N, target_size, replace=False))
            if is_sidon(A_try):
                if target_size > len(best_A):
                    best_A = A_try
                    found = True
                if found:
                    break
        if found:
            continue
        else:
            break # If we can't find size K, we can't find K+1
            
    if len(best_A) > len(A_greedy):
        h_sidon_large = np.zeros(N)
        for idx in best_A:
            h_sidon_large[idx] = 1.0
        h_sidon_large /= np.sum(h_sidon_large)
        v = _ratio(h_sidon_large)
        if v > best_val:
            best_val = v
            best_h = h_sidon_large.copy()
    
    # Local optimization from best starting point using log-space L-BFGS-B
    if time.time() < deadline:
        def neg_ratio_log(params):
            h = np.exp(params)
            h = h / np.sum(h)
            return -_ratio(h)
        
        h_pos = best_h + 1e-10
        x0 = np.log(h_pos)
        result = minimize(neg_ratio_log, x0, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14})
        h_opt = np.exp(result.x)
        h_opt = h_opt / np.sum(h_opt)
        v = _ratio(h_opt)
        if v > best_val:
            best_val = v
            best_h = h_opt.copy()
    
    # Try optimization from a few other starts
    for p in patterns[:3]:
        if time.time() >= deadline - 20:
            break
        def neg_ratio_log_p(params, p_init=p):
            h = np.exp(params)
            h = h / np.sum(h)
            return -_ratio(h)
        
        h_pos = p + 1e-10
        x0 = np.log(h_pos)
        result = minimize(neg_ratio_log_p, x0, method='L-BFGS-B',
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14})
        h_opt = np.exp(result.x)
        h_opt = h_opt / np.sum(h_opt)
        v = _ratio(h_opt)
        if v > best_val:
            best_val = v
            best_h = h_opt.copy()
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5:
        def neg_ratio_log_final(params):
            h = np.exp(params)
            h = h / np.sum(h)
            return -_ratio(h)
        
        h_pos = best_h + 1e-10
        x0 = np.log(h_pos)
        result = minimize(neg_ratio_log_final, x0, method='L-BFGS-B',
                         options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16})
        h_opt = np.exp(result.x)
        h_opt = h_opt / np.sum(h_opt)
        v = _ratio(h_opt)
        if v > best_val:
            best_val = v
            best_h = h_opt.copy()
    
    return list(best_h)
# EVOLVE-BLOCK-END
