"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

MAX_N = 10_000_000
TARGET_NUM = 3
TARGET_DEN = 2
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def generate_smooth_numbers(limit, primes):
    """Generate all numbers <= limit composed of the given primes."""
    candidates = [1]
    for p in primes:
        new_candidates = []
        for c in candidates:
            val = c
            while val * p <= limit:
                val *= p
                new_candidates.append(val)
        candidates.extend(new_candidates)
    return candidates

def sigma_smooth(n, primes):
    """Calculate sigma(n) for n composed of primes in the list."""
    result = 1
    for p in primes:
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            # sigma(p^exp) = (p^(exp+1) - 1) / (p - 1)
            result *= (p**(exp + 1) - 1) // (p - 1)
        if n == 1:
            break
    if n > 1:
        # n is a prime factor not in our list (should not happen for smooth numbers)
        result *= (n + 1)
    return result

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    # Step 1: Generate sparse candidate set of smooth numbers
    candidates = generate_smooth_numbers(MAX_N, PRIMES)
    candidates = list(set(candidates)) # Ensure uniqueness
    candidates.sort()
    
    # Step 2: Compute sigma for candidates and group
    groups = defaultdict(list)
    for n in candidates:
        if time.time() - start_time > DEADLINE:
            break
        sv = sigma_smooth(n, PRIMES)
        groups[sv].append(n)
        
    # Include n=1 if not present (sigma(1)=1)
    if 1 not in groups:
        groups[1].append(1)

    # Step 3: Find best pair
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
            lo, hi = 0, nlen - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            candidates_a = []
            for pos in [lo - 1, lo, lo + 1]:
                if 0 <= pos < nlen:
                    candidates_a.append(nums[pos])
            
            for a in candidates_a:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                if best_error_num is None:
                    best_error_num = num
                    best_error_den = den
                    best_a = a
                    best_b = b
                else:
                    if num * best_error_den < best_error_num * den:
                        best_error_num = num
                        best_error_den = den
                        best_a = a
                        best_b = b

    # Fallback if no pair found or deadline exceeded early
    if best_error_num is None:
        return [11, 6]

    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
