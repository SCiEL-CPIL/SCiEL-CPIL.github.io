import time
import math
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: return before 90s wall-clock limit. 
    # Let's aim to finish by 85s to be safe.
    deadline = 85.0
    
    # Initial known good pair from inspirations
    # sigma(31) = 32, sigma(21) = 32. |31/21 - 1.5| = 0.0238
    # sigma(122) = 186, sigma(80) = 186. |122/80 - 1.5| = |1.525 - 1.5| = 0.025
    # Let's use [31, 21] as it has a slightly better raw error.
    best_a = 31
    best_b = 21
    best_error = abs(31/21 - 1.5)
    
    # We will compute sigma up to increasing limits
    # The sieve is O(N log N). 
    # For N=2e6, it's very fast (~0.1-0.2s).
    # For N=10e6, it's slow (~30-60s).
    
    # Strategy: 
    # 1. Compute sigma up to N1 = 2e6. Search.
    # 2. If time remains, extend to N2 = 5e6. Search.
    # 3. If time remains, extend to N3 = 10e6. Search.
    
    # To extend, we just continue the sieve loop.
    # sigma array must be large enough for the max N.
    # Let's allocate for max N = 10_000_000.
    MAX_N = 10_000_000
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # We will run the sieve in chunks or just run it all at once but check time?
    # No, running the sieve for 10M takes too long.
    # Better to run the sieve up to a specific limit, then process, then extend.
    
    # However, `sigma[i::i] += i` modifies the whole array if we allocate it.
    # We can just run the loop up to `current_limit`.
    
    current_limit = 0
    
    def run_sieve_up_to(limit):
        nonlocal current_limit
        # Run sieve from current_limit + 1 to limit
        # Note: The standard sieve `for i in range(1, N+1): sigma[i::i] += i`
        # requires iterating i from 1 to N.
        # If we already did 1..2e6, we need to do 2e6+1..5e6.
        # But wait, the inner loop `sigma[i::i]` touches indices i, 2i, 3i...
        # If we stop at i=2e6, we have correctly computed sigma for all n <= 2e6?
        # Yes, because any divisor d of n <= 2e6 is <= n <= 2e6.
        # So we can safely extend.
        
        for i in range(current_limit + 1, limit + 1):
            if time.time() - start_time > deadline - 5.0:
                # Reserve 5s for processing
                return False
            sigma[i::i] += i
        current_limit = limit
        return True

    # Define search function that operates on sigma[1:current_limit]
    def find_best_pair(limit):
        nonlocal best_a, best_b, best_error
        
        if limit < 2:
            return
            
        # Extract values and indices
        # We only care about n in [1, limit]
        vals = sigma[1:limit+1]
        
        # Sort indices by sigma value
        # argsort is O(N log N). For 2M, this is fast.
        sorted_indices = np.argsort(vals)
        sorted_vals = vals[sorted_indices]
        
        # Identify groups of equal sigma values
        # We iterate through the sorted array
        current_val = sorted_vals[0]
        group_start = 0
        
        for k in range(1, limit):
            if time.time() - start_time > deadline - 2.0:
                return
                
            v = sorted_vals[k]
            
            if v != current_val:
                # Process group [group_start, k)
                group_len = k - group_start
                if group_len >= 2:
                    # Get the original numbers (1-based)
                    # sorted_indices are 0-based indices into sigma[1:limit+1]
                    # So the actual n is sorted_indices + 1
                    nums = sorted_indices[group_start:k] + 1
                    # Sort nums by value for binary search
                    nums = np.sort(nums)
                    
                    # For each b in nums, find a closest to 1.5*b
                    # We can vectorize this a bit or just loop.
                    # Looping in Python over M elements where M is group size.
                    # Most groups are small.
                    
                    # Optimization: Instead of checking every b, we can check pairs?
                    # No, just check each b and search for a.
                    
                    # To speed up, we can skip if the potential best error is already worse than current best?
                    # err = |a/b - 1.5| = |2a - 3b| / (2b)
                    # We want to minimize |2a - 3b| / (2b).
                    
                    for b in nums:
                        target = 1.5 * b
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
                            # Calculate error exactly using integers to avoid float issues?
                            # err = abs(a/b - 1.5) = abs(2a - 3b) / (2b)
                            # Compare with best_error
                            # best_error is float.
                            # Let's just use float for comparison, it's fine for this precision.
                            err = abs(a / b - 1.5)
                            if err < best_error:
                                best_error = err
                                best_a = int(a)
                                best_b = int(b)
                                
                # Start new group
                current_val = v
                group_start = k
                
        # Process last group
        group_len = limit - group_start
        if group_len >= 2:
            nums = sorted_indices[group_start:limit] + 1
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

    # Staged Search
    # Stage 1: 2M
    if run_sieve_up_to(2_000_000):
        find_best_pair(2_000_000)
        
    # Stage 2: 5M
    if time.time() - start_time < deadline - 15.0: # Reserve 15s for processing + safety
        if run_sieve_up_to(5_000_000):
            find_best_pair(5_000_000)
            
    # Stage 3: 10M
    if time.time() - start_time < deadline - 30.0: # Reserve 30s for processing + safety
        if run_sieve_up_to(10_000_000):
            find_best_pair(10_000_000)

    return [int(best_a), int(best_b)]
