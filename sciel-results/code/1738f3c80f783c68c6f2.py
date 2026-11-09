"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

MAX_N = 10_000_000
TARGET_NUM = 3
TARGET_DEN = 2

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Step 1: Compute sigma(n) for all n using a vectorized sieve approach
    # sigma(n) = sum of all positive divisors of n
    # Use the fact that sigma is multiplicative and compute via sieve
    # For efficiency, use NumPy arrays
    
    # Initialize sigma array
    sigma = np.ones(MAX_N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to all multiples of i
    # This is O(N log N) but vectorized
    # Process in chunks to manage memory and time
    for i in range(2, MAX_N + 1):
        if time.time() - start_time > DEADLINE:
            break
        # Add i to sigma[i], sigma[2i], sigma[3i], ...
        sigma[i::i] += i
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return [11, 6]

    # Step 2: Group numbers by sigma value
    # Convert to Python lists for grouping
    # Only consider n >= 2
    sigma_list = sigma[2:].tolist()
    
    groups = {}
    for idx, sv in enumerate(sigma_list):
        n = idx + 2
        if sv not in groups:
            groups[sv] = []
        groups[sv].append(n)

    # Also include n=1
    if 1 not in groups:
        groups[1] = []
    groups[1].append(1)

    # Step 3: For each group, find pair (a, b) minimizing |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    
    best_error_num = None
    best_error_den = None
    best_a = 11
    best_b = 6

    for sv, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() - start_time > DEADLINE:
            break
        nums.sort()
        nlen = len(nums)
        
        for i in range(nlen):
            b = nums[i]
            target_a = 1.5 * b
            # Binary search for position where target_a would be
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1 (and maybe lo+1)
            candidates = []
            for pos in [lo - 1, lo, lo + 1]:
                if 0 <= pos < nlen:
                    candidates.append(nums[pos])
            
            for a in candidates:
                if a == b:
                    continue
                # error = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # invalid per task
                den = 2 * b
                # Compare num/den with best_error_num/best_error_den
                if best_error_num is None:
                    best_error_num = num
                    best_error_den = den
                    best_a = a
                    best_b = b
                else:
                    # num/den < best_num/best_den  <=>  num*best_den < best_num*den
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_a = a
                        best_b = b

    # Ensure types are Python int
    best_a = int(best_a)
    best_b = int(best_b)
    
    return [best_a, best_b]

# EVOLVE-BLOCK-END
