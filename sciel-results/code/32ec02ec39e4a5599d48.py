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

def _find_max_sidon_set(n, max_k=None):
    """Find a Sidon set of maximum size in [0, n-1] using backtracking."""
    if max_k is None:
        max_k = int(np.sqrt(n)) + 2
    
    # We will search for sets of size max_k, then max_k-1, etc.
    # Backtracking search
    def backtrack(start, current_set, k):
        if len(current_set) == k:
            return current_set[:]
        
        for i in range(start, n):
            # Check if i can be added
            valid = True
            for s in current_set:
                if (s + i) in _current_sums:
                    valid = False
                    break
            if (i + i) in _current_sums:
                valid = False
            
            if valid:
                current_set.append(i)
                _current_sums.update([s + i for s in current_set[:-1]])
                _current_sums.add(i + i)
                
                res = backtrack(i + 1, current_set, k)
                if res:
                    return res
                
                # Undo
                _current_sums.difference_update([s + i for s in current_set[:-1]])
                _current_sums.discard(i + i)
                current_set.pop()
        
        return None

    # Use a global set for sums to avoid passing it around in recursion
    # But recursion depth is small (<=8), so passing is fine.
    # Let's use a helper that manages the sums set.
    
    best_set = None
    for k in range(max_k, 2, -1):
        # Try to find a set of size k
        # Simple recursive search
        def search(start, current, used_sums):
            if len(current) == k:
                return current[:]
            
            for i in range(start, n):
                # Check validity
                new_sums = set()
                valid = True
                for s in current:
                    if (s + i) in used_sums:
                        valid = False
                        break
                    new_sums.add(s + i)
                if (i + i) in used_sums:
                    valid = False
                else:
                    new_sums.add(i + i)
                
                if valid:
                    current.append(i)
                    used_sums.update(new_sums)
                    
                    res = search(i + 1, current, used_sums)
                    if res:
                        return res
                    
                    # Backtrack
                    used_sums.difference_update(new_sums)
                    current.pop()
            
            return None

        res = search(0, [], set())
        if res:
            best_set = res
            break
            
    return best_set

def construct():
    start_time = time.time()
    deadline = 110.0

    # 1. Find the best Sidon set
    sidon_set = _find_max_sidon_set(N, 8)
    
    if sidon_set is None:
        # Fallback to Uniform
        h = np.ones(N) / N
        return list(h)
        
    K = len(sidon_set)
    
    # 2. Optimize weights on the Sidon set
    # We fix the support to sidon_set.
    # We want to maximize score(h) where h is zero outside sidon_set.
    # Let h[sidon_set[j]] = w[j].
    # Score = sum(w^2) / (sum(w) * max_conv(w))
    # We can minimize the negative log score or just maximize score.
    # Using scipy.optimize.minimize to maximize score (minimize -score).
    
    def neg_score(w):
        # w are the heights on the Sidon set
        h = np.zeros(N)
        for i, idx in enumerate(sidon_set):
            h[idx] = w[i]
        # Ensure non-negative
        h = np.maximum(h, 0)
        if np.sum(h) == 0:
            return 0
        return -_score_raw(h)

    # Initial guess: uniform weights
    w0 = np.ones(K) / K
    
    # Constraints: sum(w) = 1 (scaling invariance allows this)
    # Actually, the score is scale invariant.
    # So we can fix sum(w) = 1.
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, None) for _ in range(K)]
    
    try:
        res = minimize(neg_score, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
        if res.success:
            w_opt = res.x
        else:
            w_opt = w0
    except Exception:
        w_opt = w0
        
    # Construct final h
    h_final = np.zeros(N)
    for i, idx in enumerate(sidon_set):
        h_final[idx] = w_opt[i]
        
    # Check score
    score_final = _score_raw(h_final)
    
    # Compare with uniform just in case
    h_uniform = np.ones(N) / N
    score_uniform = _score_raw(h_uniform)
    
    if score_final > score_uniform:
        return list(h_final)
    else:
        return list(h_uniform)
