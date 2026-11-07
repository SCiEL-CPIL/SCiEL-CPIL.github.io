import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin for 60s limit
    
    N = 10_000_000
    
    # Fallback pair from inspiration 2 (score 14.0)
    # sigma(31) = 1 + 31 = 32
    # sigma(21) = 1 + 3 + 7 + 21 = 32
    # |31/21 - 1.5| = |1.476... - 1.5| = 0.0238...
    # Score = 1/0.0238... = 42? No, wait.
    # Let's check the score of [31, 21].
    # 31/21 = 1.47619
    # 1.5 - 1.47619 = 0.0238095
    # 1 / 0.0238095 = 42.0
    # The inspiration says score 14.0. Maybe I miscalculated or the score definition is different?
    # "The score is the binary64 serialization of that rational... seed/value... 1.0 equals seed"
    # If seed is [11, 6]: sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = |1.833 - 1.5| = 0.333.
    # 1/0.333 = 3.0.
    # If score is seed/value, and seed score is 1.0, then value for [11,6] is 1.0? No, "raw value is float(|a/b-3/2|)".
    # "search score is seed/value". If seed program returns [11,6], its raw value is 0.333.
    # If score = 1/raw_value, then [11,6] score is 3.
    # But the prompt says "CURRENT PROGRAM (score 1.0000): return [11,6]".
    # This implies the scoring mechanism might be different or [11,6] is considered the baseline "1.0".
    # Actually, if the score is `seed / value`, and seed is the value of the seed program, then:
    # Seed value = |11/6 - 1.5| = 1/3.
    # Score = (1/3) / (1/3) = 1.0.
    # So for [31, 21], value = |31/21 - 1.5| = |1.47619 - 1.5| = 0.0238095.
    # Score = (1/3) / 0.0238095 = 0.3333 / 0.0238095 = 14.0.
    # Yes, this matches inspiration 2's score of 14.0.
    
    best_a = 31
    best_b = 21
    best_error = abs(31/21 - 1.5)
    
    try:
        # 1. Compute sigma(n) using NumPy
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Vectorized sieve
        # This loop is the main cost.
        for i in range(1, N + 1):
            if time.time() - start_time > deadline:
                break
            sigma[i::i] += i
            
        if time.time() - start_time > deadline:
            return [best_a, best_b]

        # 2. Find groups with count >= 2
        vals = sigma[1:]
        
        # Sort indices to group by sigma value
        sorted_indices = np.argsort(vals)
        sorted_vals = vals[sorted_indices]
        
        # Identify groups
        # We iterate through the sorted array to find contiguous segments of equal values
        current_val = sorted_vals[0]
        group_start = 0
        
        for k in range(1, N):
            if time.time() - start_time > deadline:
                break
                
            v = sorted_vals[k]
            
            if v != current_val:
                # Process the previous group [group_start, k)
                group_len = k - group_start
                if group_len >= 2:
                    # Get the original numbers (1-based)
                    # sorted_indices are 0-based indices into sigma[1:], so add 1
                    nums = sorted_indices[group_start:k] + 1
                    
                    # Sort nums by value for binary search
                    nums = np.sort(nums)
                    
                    # For each b in nums, find a closest to 1.5*b
                    # To avoid O(M^2) for large groups, we can limit checks or use binary search
                    # M is typically small, but can be large for small sigma values.
                    # Let's just iterate and use searchsorted
                    for b in nums:
                        target = 1.5 * b
                        # Find insertion point for target in nums
                        idx = np.searchsorted(nums, target)
                        
                        # Check idx and idx-1
                        candidates = []
                        if idx < len(nums):
                            candidates.append(nums[idx])
                        if idx > 0:
                            candidates.append(nums[idx-1])
                            
                        for a in candidates:
                            if a == b:
                                continue
                            err = abs(a / b - 1.5)
                            if err < best_error:
                                best_error = err
                                best_a = int(a)
                                best_b = int(b)
                                
                # Start new group
                current_val = v
                group_start = k
                
        # Process the last group
        group_len = N - group_start
        if group_len >= 2:
            nums = sorted_indices[group_start:N] + 1
            nums = np.sort(nums)
            for b in nums:
                target = 1.5 * b
                idx = np.searchsorted(nums, target)
                
                candidates = []
                if idx < len(nums):
                    candidates.append(nums[idx])
                if idx > 0:
                    candidates.append(nums[idx-1])
                    
                for a in candidates:
                    if a == b:
                        continue
                    err = abs(a / b - 1.5)
                    if err < best_error:
                        best_error = err
                        best_a = int(a)
                        best_b = int(b)

    except Exception:
        pass

    return [best_a, best_b]
