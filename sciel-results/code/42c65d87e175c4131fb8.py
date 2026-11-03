import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _is_sidon_incremental(current_sums, current_set, new_elem):
    """
    Check if adding new_elem to current_set maintains the Sidon property.
    current_sums: set of sums i+j for i<=j in current_set
    current_set: list of elements in current_set
    """
    for e in current_set:
        s = new_elem + e
        if s in current_sums:
            return False, current_sums
        # Add new sums involving new_elem
        # We will update current_sums if this returns True
    # Also need to check new_elem + new_elem
    s_self = new_elem + new_elem
    if s_self in current_sums:
        return False, current_sums
    
    # If we get here, it's valid. Update sums.
    new_sums = set(current_sums)
    for e in current_set:
        new_sums.add(new_elem + e)
    new_sums.add(s_self)
    return True, new_sums

def _find_sidon_set(n, k, max_tries=5000):
    """
    Find a Sidon set of size k in {0, ..., n-1}.
    Returns the set as a list of indices, or None.
    """
    for _ in range(max_tries):
        # Random order of candidates
        candidates = np.arange(n)
        np.random.shuffle(candidates)
        
        current_set = []
        current_sums = set()
        
        for c in candidates:
            if len(current_set) == k:
                break
            valid, new_sums = _is_sidon_incremental(current_sums, current_set, c)
            if valid:
                current_set.append(c)
                current_sums = new_sums
        
        if len(current_set) == k:
            return current_set
    return None

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin for 120s limit
    
    # Initial best: Uniform
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Try to find Sidon sets of decreasing size
    # Max possible is around sqrt(50) ~ 7.
    # We try 9, 8, 7, 6, 5
    for k in range(9, 4, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple times to find a set of this size
        for attempt in range(2000):
            if time.time() > deadline:
                break
            s = _find_sidon_set(N, k, max_tries=1)
            if s is not None:
                # Construct h
                h = np.zeros(N)
                for idx in s:
                    h[idx] = 1.0
                h /= np.sum(h)
                score = _score_raw(h)
                if score > best_score:
                    best_score = score
                    best_h = h.copy()
                    found = True
                    # If we found a good one, we might not need to try more for this k
                    # but a different set might allow better weight optimization?
                    # For now, uniform is optimal for Sidon.
                    break
        
        # If we found a set of size k, the score should be approx k.
        # If k=8, score ~8. If k=7, score ~7.
        # We can stop if we have a high score.
        if best_score > k - 0.5:
            # Found a set of size k (or larger effectively)
            # We can break if we have a very high score
            if best_score > 7:
                break

    # Polish the best candidate with L-BFGS
    # This might help if the optimal weights are not exactly uniform
    # or if the Sidon structure allows slight deviations that improve the ratio
    if time.time() < deadline - 10:
        h_start = best_h.copy()
        # Ensure no zeros for log
        h_start[h_start < 1e-10] = 1e-10
        x0 = np.log(h_start)
        
        def neg_obj(x):
            h = np.exp(x)
            h = h / np.sum(h)
            l2_sq = np.sum(h**2)
            h_conv = np.convolve(h, h, mode='full')
            max_conv = np.max(h_conv)
            if max_conv == 0: return 0.0
            return -(l2_sq / max_conv)
        
        try:
            res = minimize(neg_obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10})
            if res.success or res.fun < 0:
                h_candidate = np.exp(res.x)
                h_candidate /= np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Final check
    if time.time() < deadline - 1:
         # One more polish
         h_start = best_h.copy()
         h_start[h_start < 1e-10] = 1e-10
         x0 = np.log(h_start)
         
         def neg_obj(x):
             h = np.exp(x)
             h = h / np.sum(h)
             l2_sq = np.sum(h**2)
             h_conv = np.convolve(h, h, mode='full')
             max_conv = np.max(h_conv)
             if max_conv == 0: return 0.0
             return -(l2_sq / max_conv)
         
         try:
             res = minimize(neg_obj, x0, method='L-BFGS-B', 
                           options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12})
             if res.success or res.fun < 0:
                 h_candidate = np.exp(res.x)
                 h_candidate /= np.sum(h_candidate)
                 s_candidate = _score_raw(h_candidate)
                 if s_candidate > best_score:
                     best_score = s_candidate
                     best_h = h_candidate.copy()
         except Exception:
             pass

    return list(best_h)
