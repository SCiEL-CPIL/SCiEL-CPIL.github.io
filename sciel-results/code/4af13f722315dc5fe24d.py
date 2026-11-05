import time
import math
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 30
    
    # Compute LCM(1..30)
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Scaled weights: w[i] = L // i for i in 1..n
    # We want to minimize |sum(c_i * w_i)| / L
    # So we minimize |sum(c_i * w_i)| where w_i are integers.
    w = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    half = n // 2
    w_left = w[:half]
    w_right = w[half:]
    
    def enumerate_sums_numpy(weights):
        """
        Enumerate all sums for given weights with coeffs in {-1,0,1}.
        Returns a numpy array of sums.
        """
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        for w_i in weights:
            # New sums: old_sums, old_sums + w_i, old_sums - w_i
            # This can be done with concatenation
            new_sums = np.concatenate([sums, sums + w_i, sums - w_i])
            sums = new_sums
        return sums

    # Check time
    if (time.time() - start_time) > deadline - 40:
        # Fallback
        return [0] * 28 + [1, -1]

    # Enumerate left sums
    left_sums = enumerate_sums_numpy(w_left)
    
    if (time.time() - start_time) > deadline - 30:
        return [0] * 28 + [1, -1]

    # Enumerate right sums
    right_sums = enumerate_sums_numpy(w_right)
    
    if (time.time() - start_time) > deadline - 10:
        return [0] * 28 + [1, -1]

    # Sort right sums for binary search
    right_sums_sorted = np.sort(right_sums)
    
    best_abs_N = None
    best_left_sum = None
    best_right_sum = None
    
    # Iterate through left sums
    # To speed up, we can iterate in chunks or just directly.
    # 3^15 ~ 14M iterations. In Python, this might be slow.
    # Let's optimize the loop.
    
    # Instead of a Python loop over 14M items, can we vectorize?
    # Not easily for binary search with variable targets.
    # But we can try to process in batches or just ensure the loop is tight.
    
    # 14M iterations in Python might take ~10-20 seconds.
    # Let's try to be efficient.
    
    # We can also prune: if we find a sum of 1 (scaled), that's likely the minimum possible non-zero integer sum?
    # The minimum non-zero integer sum is 1.
    # If we find |N| == 1, we can stop immediately as it's the global minimum.
    
    found_min = False
    # We need to map the sum back to coefficients.
    # For now, just find the best sums.
    
    # To handle the 14M loop efficiently:
    # We can use numpy's searchsorted on the whole array? No, targets are different.
    
    # Let's use a standard loop but break early if we find 1.
    
    for ls in left_sums:
        if (time.time() - start_time) > deadline - 5:
            break
            
        target = -ls
        pos = np.searchsorted(right_sums_sorted, target)
        
        # Check pos and pos-1
        # Also check pos+1, pos-2 just in case of duplicates or edge cases?
        # searchsorted returns the first index where target can be inserted.
        # So right_sums_sorted[pos-1] <= target < right_sums_sorted[pos] (if pos>0)
        
        candidates = []
        if pos > 0:
            candidates.append(right_sums_sorted[pos-1])
        if pos < len(right_sums_sorted):
            candidates.append(right_sums_sorted[pos])
            
        # Add neighbors to be safe against floating point/int issues? No, it's exact int.
        # But if there are duplicates, we might miss a closer one?
        # searchsorted finds the boundary. The closest value is either at pos or pos-1.
        
        for rs in candidates:
            total = ls + rs
            if total == 0:
                continue
            abs_total = abs(total)
            
            if best_abs_N is None or abs_total < best_abs_N:
                best_abs_N = abs_total
                best_left_sum = ls
                best_right_sum = rs
                
                if best_abs_N == 1:
                    found_min = True
                    break
                    
        if found_min:
            break

    if best_abs_N is None:
        # Should not happen, but fallback
        return [0] * 28 + [1, -1]

    # Reconstruct coefficients
    def find_coeffs(weights, target):
        """
        Find coefficients in {-1,0,1} such that sum(c_i * w_i) = target.
        Uses DFS with pruning.
        """
        n_w = len(weights)
        # Precompute suffix sums of absolute weights for pruning
        suffix_max = np.zeros(n_w + 1, dtype=np.int64)
        for i in range(n_w - 1, -1, -1):
            suffix_max[i] = suffix_max[i+1] + abs(weights[i])
        
        def backtrack(idx, current_sum, coeffs):
            if idx == n_w:
                if current_sum == target:
                    return coeffs.copy()
                return None
            
            # Pruning
            if abs(target - current_sum) > suffix_max[idx]:
                return None
            
            # Try 0, 1, -1
            # Order might matter for speed, but any order works for correctness.
            # Trying 0 first might find solutions with fewer non-zeros?
            
            for c in (0, 1, -1):
                coeffs[idx] = c
                result = backtrack(idx + 1, current_sum + c * weights[idx], coeffs)
                if result is not None:
                    return result
            
            coeffs[idx] = 0 # Reset
            return None
        
        return backtrack(0, 0, np.zeros(n_w, dtype=np.int8))

    coeffs_left = find_coeffs(w_left, best_left_sum)
    coeffs_right = find_coeffs(w_right, best_right_sum)
    
    if coeffs_left is None or coeffs_right is None:
        # Fallback if reconstruction fails (shouldn't happen)
        return [0] * 28 + [1, -1]
    
    full_coeffs = list(coeffs_left) + list(coeffs_right)
    
    # Verify
    total = sum(c * w[i] for i, c in enumerate(full_coeffs))
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return full_coeffs

