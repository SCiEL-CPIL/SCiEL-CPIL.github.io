import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin
    N = 10000000

    # 1. Linear sieve for sigma
    sigma = np.zeros(N + 1, dtype=np.int64)
    sp = np.zeros(N + 1, dtype=np.int64)
    pe = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    primes = []

    # Optimization: Use Python lists for primes, numpy for arrays
    # The loop is the bottleneck.
    for i in range(2, N + 1):
        if time.time() - start_time > DEADLINE - 10:
            # If we run out of time during sieve, we can't proceed correctly.
            # But 10^7 sieve is fast (~1s).
            break
        if sp[i] == 0:
            sp[i] = i
            pe[i] = i
            sigma[i] = 1 + i
            primes.append(i)
        for p in primes:
            if p > sp[i] or i * p > N:
                break
            sp[i * p] = p
            if i % p == 0:
                pe[i * p] = pe[i] * p
                m = i // pe[i]
                sigma[i * p] = sigma[i] + sigma[m] * pe[i] * p
                break
            else:
                pe[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)

    if time.time() - start_time > DEADLINE - 5:
        # Fallback to a known small pair if sieve failed or is too slow
        return [11, 6]

    # 2. Group numbers by sigma value
    # sigma values can be large. Use a dictionary.
    groups = defaultdict(list)
    for i in range(1, N + 1):
        groups[sigma[i]].append(i)

    # 3. Search for best pair
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Initialize with known pair
    # sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = |1.833 - 1.5| = 0.333
    best_error = abs(11/6 - 1.5)
    
    # Iterate through groups
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        if time.time() - start_time > DEADLINE - 2:
            break
            
        # Sort the numbers in this group
        nums.sort()
        
        # For each b in the group, find best a
        # We can iterate b from largest to smallest to potentially find large b early?
        # Or just iterate all.
        # To prune: if 1/(2*b) >= best_error, then even k=1 is not good enough.
        # So b < 1/(2*best_error). We can skip small b.
        min_b_needed = int(1.0 / (2.0 * best_error)) + 1
        
        for b in nums:
            if b < min_b_needed:
                continue
                
            # Target a is 1.5 * b
            target_a = 1.5 * b
            
            # Binary search for target_a in nums
            # Use numpy searchsorted? Or bisect?
            # bisect is in standard library and fast.
            import bisect
            idx = bisect.bisect_left(nums, target_a)
            
            # Candidates: idx-1, idx, idx+1 (just in case, though convexity suggests 2 is enough)
            # Actually, for |2a - 3b|, the minimum is at a=1.5b.
            # The closest integer in the set is either floor or ceil in the set.
            # So idx and idx-1 are sufficient.
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                if a > N or a < 1:
                    continue
                
                # Calculate error
                # |a/b - 3/2| = |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue # Invalid per contract
                
                err = diff / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    # Update min_b_needed to prune more
                    min_b_needed = int(1.0 / (2.0 * best_error)) + 1

    return [int(best_a), int(best_b)]
