import numpy as np
import time
import itertools

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
        max_k = int(np.sqrt(n)) + 4  # Slightly higher upper bound
    
    def backtrack(start, current_set, used_sums, k):
        if len(current_set) == k:
            return current_set[:]
        
        # Pruning: if not enough elements left to reach k
        if n - start < k - len(current_set):
            return None
            
        for i in range(start, n):
            # Check if i can be added without creating duplicate sums
            valid = True
            # Optimization: check sums with existing elements
            for s in current_set:
                if (s + i) in used_sums:
                    valid = False
                    break
            if not valid:
                continue
            
            if (i + i) in used_sums:
                continue
            
            # Add i to the set and update sums
            current_set.append(i)
            new_sums = set()
            for s in current_set[:-1]:
                new_sums.add(s + i)
            new_sums.add(i + i)
            
            used_sums.update(new_sums)
            
            res = backtrack(i + 1, current_set, used_sums, k)
            if res:
                return res
            
            # Backtrack
            used_sums.difference_update(new_sums)
            current_set.pop()
        
        return None

    for k in range(max_k, 2, -1):
        res = backtrack(0, [], set(), k)
        if res:
            return res
            
    return None

def construct():
    # 1. Find the largest Sidon set
    # Try to find a set of size 10, then 9, etc.
    sidon_set = _find_max_sidon_set(N, 10)
    
    if sidon_set is None:
        # Fallback to Uniform
        h = np.ones(N) / N
        return list(h)
        
    K = len(sidon_set)
    
    # 2. Use uniform weights, which are optimal for Sidon sets
    h_final = np.zeros(N)
    for idx in sidon_set:
        h_final[idx] = 1.0 / K
        
    return list(h_final)
