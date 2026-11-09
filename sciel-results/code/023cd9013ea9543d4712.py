"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of numbers we need: n+i for n in [1000, 1000000], i in [1, 24]
    # So numbers range from 1001 to 1000024
    lo = 1001
    hi = 1000024
    size = hi - lo + 1
    
    # Compute small-prime factor for each number in [lo, hi]
    # small_factor[m] = product of p^v where p in primes and p^v divides m
    # Equivalently: divide m by all factors of primes >= 29... 
    # Better: for each prime p in primes, compute the p-part of each number
    
    # Create array of numbers
    nums = np.arange(lo, hi + 1, dtype=np.int64)
    
    # small_factor starts as 1 for all
    small_factor = np.ones(size, dtype=np.int64)
    
    for p in primes:
        # For each number, extract the p-part
        # p-part of m = largest p^k dividing m
        # We can do this iteratively: while divisible by p, divide and multiply
        # Vectorized approach:
        # temp = nums
        # for _ in range(max_power):
        #     mask = temp % p == 0
        #     temp[mask] //= p
        #     small_factor[mask] *= p
        
        # Max power: for p=2, 2^20 > 1000024, so at most 20 iterations
        # For p=23, at most 5 iterations
        temp = nums.copy()
        for _ in range(20):  # enough for p=2
            mask = temp % p == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            small_factor[mask] *= p
    
    # Now small_factor[i] is the small-prime factor of (lo + i)
    # We need to find n in [1000, 1000000] such that the number of distinct
    # values in small_factor[n - lo + 1 : n - lo + 1 + 24] is minimized
    # n+i ranges from n+1 to n+24, so indices in small_factor are (n+1-lo) to (n+24-lo)
    # n+1-lo = n+1-1001 = n-1000
    # n+24-lo = n+24-1001 = n-977
    # So window is small_factor[n-1000 : n-977+1] = small_factor[n-1000 : n-976]
    # Length = (n-976) - (n-1000) = 24. Correct.
    
    # n ranges from 1000 to 1000000
    # Index start = n - 1000, ranges from 0 to 999000
    # Index end = n - 976, ranges from 24 to 999024
    # We need end < size, so n - 976 < size => n < size + 976 = 1000024 + 976 = 1001000
    # But n max is 1000000, so n - 976 = 999024 < 1000024. OK.
    
    # Sliding window distinct count
    # Window size = 24
    # For each position, count distinct values in the window
    
    # Approach: use a dictionary to track counts, slide the window
    # But that's O(N) with dict ops. Let's try it - N ~ 1M, should be fast enough.
    
    W = 24
    N = 1000000 - 1000 + 1  # number of valid n values = 999001
    
    # Use a frequency dict for the sliding window
    best_n = 1000
    best_distinct = float('inf')
    
    # Initialize window for n=1000: indices 0 to 23
    freq = defaultdict(int)
    distinct_count = 0
    
    # First window
    for i in range(W):
        val = small_factor[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
    
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000
    
    # Slide window
    # n = 1000 + s, s from 1 to 999000
    # Window: indices s to s+23
    for s in range(1, N):
        # Remove element at index s-1
        old_val = small_factor[s - 1]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            distinct_count -= 1
        
        # Add element at index s + W - 1
        new_val = small_factor[s + W - 1]
        if freq[new_val] == 0:
            distinct_count += 1
        freq[new_val] += 1
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = 1000 + s
        
        # Check time
        if s % 100000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
