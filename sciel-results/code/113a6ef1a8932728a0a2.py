import numpy as np
import time
from math import lcm

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
    
    def gen_sums_numpy(w):
        sums = np.array([0], dtype=np.int64)
        for weight in w:
            # Create three versions: -w, 0, +w added to current sums
            # Using broadcasting or simple addition
            new_sums = np.concatenate([
                sums - weight,
                sums,
                sums + weight
            ])
            sums = new_sums
        return sums

    # Generate all possible sums for the two halves
    # This is the most memory and time intensive part
    sums1 = gen_sums_numpy(w1)
    sums2 = gen_sums_numpy(w2)
    
    # Sort the sums
    sums1.sort()
    sums2.sort()
    
    # Two-pointer search for minimum non-zero absolute sum
    # We want to minimize |s1 + s2| subject to s1 + s2 != 0
    i = 0
    j = len(sums2) - 1
    best_abs = float('inf')
    best_s1 = None
    best_s2 = None
    
    # To speed up, we can use binary search if the arrays are static, 
    # but two-pointer is O(N) which is optimal for sorted arrays.
    # However, Python loops are slow. 14 million iterations might be too slow in pure Python.
    # Let's check the size. 14 million iterations in Python takes ~1-2 seconds? 
    # Actually, a tight Python loop of 14M iterations takes ~1s. 
    # The logic inside is simple comparisons. It should fit in 55s.
    
    while i < len(sums1) and j >= 0:
        if time.time() > deadline:
            break
            
        s1 = sums1[i]
        s2 = sums2[j]
        total = s1 + s2
        
        if total == 0:
            # We must skip 0. 
            # Check neighbors: (i+1, j) and (i, j-1)
            # Since sums are sorted, moving i up increases sum, moving j down decreases sum.
            # We want the smallest non-zero absolute value.
            
            # Candidate 1: i+1, j
            if i + 1 < len(sums1):
                nt = sums1[i+1] + s2
                if nt != 0 and abs(nt) < best_abs:
                    best_abs = abs(nt)
                    best_s1 = sums1[i+1]
                    best_s2 = s2
            
            # Candidate 2: i, j-1
            if j - 1 >= 0:
                nt = s1 + sums2[j-1]
                if nt != 0 and abs(nt) < best_abs:
                    best_abs = abs(nt)
                    best_s1 = s1
                    best_s2 = sums2[j-1]
            
            i += 1
            j -= 1
        elif total > 0:
            if abs(total) < best_abs:
                best_abs = abs(total)
                best_s1 = s1
                best_s2 = s2
            # To reduce total, we need to decrease s2 (since s1 is fixed for now)
            j -= 1
        else:
            if abs(total) < best_abs:
                best_abs = abs(total)
                best_s1 = s1
                best_s2 = s2
            # To increase total (closer to 0 from negative), we need to increase s1
            i += 1

    if best_s1 is None:
        # Fallback: return a valid non-zero sum
        # e.g., c_1 = 1, others 0 -> sum = 1
        return [1] + [0]*29

    # Backtrack to find coefficients
    def backtrack(w, target):
        k = len(w)
        # Precompute min and max sums for suffixes
        # suffix_min[i] = min sum using w[i:]
        # suffix_max[i] = max sum using w[i:]
        suffix_min = [0] * (k + 1)
        suffix_max = [0] * (k + 1)
        for i in range(k - 1, -1, -1):
            suffix_min[i] = suffix_min[i+1] - w[i]
            suffix_max[i] = suffix_max[i+1] + w[i]
        
        coeffs = [0] * k
        current_sum = 0
        
        def dfs(idx, current_sum):
            if idx == k:
                return current_sum == target
            
            # Pruning
            # If current_sum + suffix_min[idx+1] > target or current_sum + suffix_max[idx+1] < target
            # Wait, we are at idx, so we pick c_idx, then go to idx+1.
            # The remaining sum must be target - (current_sum + c_idx * w[idx]).
            # The min possible sum from idx+1 to end is suffix_min[idx+1].
            # The max possible sum from idx+1 to end is suffix_max[idx+1].
            
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
