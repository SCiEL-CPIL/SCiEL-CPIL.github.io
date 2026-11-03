import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * AutoConvMax)."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums are distinct)."""
    sums = set()
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _find_max_sidon_set(n, max_k=10):
    """Find a Sidon set of maximum size in [0, n-1] using backtracking with heuristics."""
    
    def backtrack(start, current_set, used_sums, k):
        if len(current_set) == k:
            return current_set[:]
        
        # Heuristic: Try indices that are further apart first? Or just standard.
        for i in range(start, n):
            # Check if i can be added
            valid = True
            for s in current_set:
                if (s + i) in used_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            if (i + i) in used_sums:
                continue
            
            current_set.append(i)
            new_sums = set()
            for s in current_set[:-1]:
                new_sums.add(s + i)
            new_sums.add(i + i)
            
            used_sums.update(new_sums)
            
            res = backtrack(i + 1, current_set, used_sums, k)
            if res:
                return res
            
            used_sums.difference_update(new_sums)
            current_set.pop()
        
        return None

    for k in range(max_k, 2, -1):
        res = backtrack(0, [], set(), k)
        if res:
            return res
            
    return None

def _optimize_weights(indices):
    """Optimize weights for a given set of indices to minimize peak autoconvolution."""
    K = len(indices)
    if K == 0:
        return np.zeros(N)
    
    # Initial guess: uniform
    h0 = np.zeros(N)
    for idx in indices:
        h0[idx] = 1.0 / K
        
    def objective(x):
        # x are the weights for the indices in 'indices'
        # Reconstruct full h
        h = np.zeros(N)
        for i, idx in enumerate(indices):
            h[idx] = x[i]
        
        # Calculate autoconvolution peak
        # Note: x sums to 1, x >= 0
        # We want to minimize max(h*h)
        h_conv = np.convolve(h, h, mode='full')
        return np.max(h_conv)
        
    # Constraints: sum(x) = 1, x >= 0
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    bounds = [(0, None) for _ in range(K)]
    
    try:
        res = minimize(objective, h0[indices], method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-9})
        if res.success:
            h_opt = np.zeros(N)
            for i, idx in enumerate(indices):
                h_opt[idx] = res.x[i]
            # Normalize just in case
            h_opt /= np.sum(h_opt)
            return h_opt
    except Exception:
        pass
        
    return h0

def construct():
    # 1. Find the largest Sidon set
    sidon_set = _find_max_sidon_set(N, 9)
    
    if sidon_set is None:
        # Fallback to Uniform
        h = np.ones(N) / N
        return list(h)
        
    # 2. Optimize weights for this set
    h_final = _optimize_weights(sidon_set)
        
    return list(h_final)
