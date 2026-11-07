"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
MAX_N = 10_000_000
TARGET_NUM = 3
TARGET_DEN = 2

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin within 60s limit
    
    # Step 1: Sieve to compute sigma(n) for all n up to MAX_N
    # sigma(n) = sum of divisors of n
    # Use the multiplicative property: for each prime p, update sigma for multiples
    
    # Initialize sigma array
    sigma = [0] * (MAX_N + 1)
    sigma[0] = 0
    # For n >= 1, sigma[n] will be computed
    
    # Use a modified sieve: 
    # sigma is multiplicative. We can compute it by iterating over primes.
    # Alternative: for each i from 1 to MAX_N, add i to sigma[j] for all j that are multiples of i
    # This is O(n log n) which for n=10^7 is about 10^7 * ln(10^7) ≈ 10^7 * 16 ≈ 1.6 * 10^8 operations
    # This might be too slow in Python. Let's use a smarter approach.
    
    # Better approach: factorize using smallest prime factor (SPF) sieve, then compute sigma multiplicatively
    # SPF sieve: O(n log log n)
    
    # Build SPF array
    spf = list(range(MAX_N + 1))  # smallest prime factor
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute sigma(n) for all n using SPF
    # sigma is multiplicative: if n = p1^a1 * p2^a2 * ... then sigma(n) = prod((p^(a+1)-1)/(p-1))
    
    sigma_arr = [1] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            # p divides m, so n has p^k where k >= 2
            # We need to track the power of p in n
            # Better: precompute for each n, the highest power of spf[n] dividing n
            pass
    
    # Let me use a different approach: iterate and build sigma incrementally
    # For each n, factor using spf and compute sigma
    
    # Actually, let's use the divisor-sum sieve approach but more carefully
    # sigma[n] = sum_{d|n} d
    # We can compute this by: for d from 1 to MAX_N, for multiple m of d, sigma[m] += d
    # This is O(n * H_n) where H_n is harmonic number ~ n log n
    # For n = 10^7, this is ~ 10^7 * 16 = 1.6 * 10^8 additions
    # In Python this will be too slow.
    
    # Let's use NumPy for speed
    import numpy as np
    
    sigma_np = np.zeros(MAX_N + 1, dtype=np.int64)
    
    for d in range(1, MAX_N + 1):
        if time.time() > deadline - 5:
            break
        # Add d to all multiples of d
        multiples = np.arange(d, MAX_N + 1, d)
        sigma_np[multiples] += d
    
    # Convert to list for easier handling
    sigma_list = sigma_np.tolist()
    
    # Step 2: Group numbers by sigma value
    # Use a dictionary: sigma_value -> list of numbers with that sigma
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = sigma_list[n]
        sigma_groups[s].append(n)
        if time.time() > deadline - 3:
            break
    
    # Step 3: For each group, find pair (a,b) minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this. For a given group, we want a/b close to 3/2, i.e., a close to 3b/2
    
    best_error = float('inf')
    best_pair = (11, 6)  # fallback to seed
    
    for s, nums in sigma_groups.items():
        if time.time() > deadline - 2:
            break
        if len(nums) < 2:
            continue
        
        # Sort nums for binary search
        nums_sorted = sorted(nums)
        
        # For each b in nums_sorted, find a in nums_sorted such that a ≈ 3b/2
        # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
        # For fixed b, we want a close to 3b/2
        # Use binary search to find candidates near 3b/2
        
        for b in nums_sorted:
            target_a = 3 * b / 2
            # Binary search for target_a in nums_sorted
            idx = bisect.bisect_left(nums_sorted, int(target_a))
            
            # Check candidates around idx
            candidates = []
            for i in [idx - 2, idx - 1, idx, idx + 1, idx + 2]:
                if 0 <= i < len(nums_sorted):
                    candidates.append(nums_sorted[i])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| exactly as rational
                # |a/b - 3/2| = |2a - 3b| / (2b)
                numerator = abs(2 * a - 3 * b)
                denominator = 2 * b
                if numerator == 0:
                    continue  # raw value of 0 is invalid
                error = numerator / denominator
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
    
    # Ensure we return Python ints in a list
    a, b = best_pair
    return [int(a), int(b)]

import bisect
# EVOLVE-BLOCK-END
