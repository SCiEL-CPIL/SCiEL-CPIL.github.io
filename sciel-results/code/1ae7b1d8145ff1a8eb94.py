import numpy as np
import time
from math import lcm
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin out of 60s
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    weights = [L // i for i in range(1, n + 1)]
    
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    def gen_sums(w):
        sums = [0]
        for weight in w:
            new_sums = []
            # Optimization: use list comprehension for speed
            # sums is a list of ints
            new_sums = [s - weight for s in sums] + sums + [s + weight for s in sums]
            sums = new_sums
        return sums

    # Generate sums for first half
    sums1 = gen_sums(w1)
    
    # Sort sums1 for binary search
    sums1.sort()
    
    # Generate sums for second half
    sums2 = gen_sums(w2)
    
    best_abs = float('inf')
    best_s1 = None
    best_s2 = None
    
    # Iterate over sums2
    # We can potentially break early if we find 1
    for s2 in sums2:
        if time.time() > deadline:
            break
            
        target = -s2
        # Find the position where target would be inserted
        pos = bisect_left(sums1, target)
        
        # Check pos
        if pos < len(sums1):
            s1 = sums1[pos]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        break
        
        # Check pos - 1
        if pos > 0:
            s1 = sums1[pos - 1]
            total = s1 + s2
            if total != 0:
                abs_total = abs(total)
                if abs_total < best_abs:
                    best_abs = abs_total
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs == 1:
                        break

    if best_s1 is None:
        # Fallback: return a valid non-zero sum
        # e.g., c_1 = 1, others 0 -> sum = 1
        return [1] + [0]*29

    # Backtrack to find coefficients
    # We need to find c1, ..., c15 such that sum(c_i * w1[i]) = best_s1
    # and c16, ..., c30 such that sum(c_i * w2[i-15]) = best_s2
    
    def backtrack(w, target):
        k = len(w)
        # Precompute min and max sums for suffixes to prune
        suffix_min = [0] * (k + 1)
        suffix_max = [0] * (k + 1)
        for i in range(k - 1, -1, -1):
            suffix_min[i] = suffix_min[i+1] - w[i]
            suffix_max[i] = suffix_max[i+1] + w[i]
        
        coeffs = [0] * k
        
        def dfs(idx, current_sum):
            if idx == k:
                return current_sum == target
            
            # Pruning
            # We need to pick c in {-1, 0, 1}
            # such that target - (current_sum + c*w[idx]) is reachable by suffix[idx+1]
            
            for c in [-1, 0, 1]:
                new_sum = current_sum + c * w[idx]
                # Check if target is reachable from new_sum with remaining weights
                if new_sum + suffix_min[idx+1] <= target <= new_sum + suffix_max[idx+1]:
                    coeffs[idx] = c
                    if dfs(idx + 1, new_sum):
                        return True
            return False
        
        if dfs(0, 0):
            return coeffs
        else:
            # Should not happen if sums were generated correctly
            return [1] + [0]*(k-1)

    c1 = backtrack(w1, best_s1)
    c2 = backtrack(w2, best_s2)
    
    result = c1 + c2
    
    # Ensure types are standard Python ints
    return [int(x) for x in result]
