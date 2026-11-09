import time
import numpy as np
from bisect import bisect_left

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma using NumPy vectorized sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
    
    if time.time() > deadline - 10:
        return [11, 6]

    # 2. Build a dictionary mapping sigma value -> sorted list of n values
    # This avoids the expensive O(n log n) argsort
    # We'll process by iterating over b and checking nearby a values
    
    # For efficiency, build a dict: sigma_value -> list of n
    # But with 10^7 entries, we need to be careful about memory
    
    # Alternative: iterate over b from 1 to MAX_N, compute target a range,
    # and check sigma equality directly using the sigma array
    
    best_num = 1  # numerator of error fraction
    best_den = 3  # denominator: error = 1/3 for [11,6]
    best_pair = (11, 6)
    
    # For each b, a should be near 3b/2
    # |2a - 3b| / (2b) is minimized when |2a - 3b| is small
    # For a given b, try a values where 2a - 3b is in some small range
    
    # Key insight: if we find |2a - 3b| = 1, that's optimal for that b
    # We want to find the smallest possible |2a - 3b| over all valid pairs
    
    # Strategy: iterate over b, for each b check a values near 3b/2
    # But we need sigma(a) == sigma(b)
    
    # To speed this up, build a lookup: for each sigma value, what n values have it?
    # Use a dictionary approach with numpy
    
    # Actually, let's use a smarter approach:
    # For each possible value of |2a - 3b| = k (starting from 1),
    # check if there exists a pair with that k and sigma(a) == sigma(b)
    
    # But that's complex. Let's stick with iterating b and checking nearby a.
    
    # Optimization: for each b, the best a is the one closest to 3b/2 with sigma(a) == sigma(b)
    # We can check a few candidates around 3b/2
    
    # Let's iterate b from 1 to MAX_N
    # For each b, check a in a small window around 3b/2
    
    # To make this fast, we'll batch-process using numpy
    # For each b, target_a = 3b//2, check a in [target_a - W, target_a + W]
    
    # But checking each b individually in Python is slow for 10^7 iterations
    # Let's use a different approach: vectorize the check
    
    # For a given offset d = 2a - 3b, we want d to be small
    # d = 2a - 3b => a = (3b + d) / 2
    # For d to make a integer, 3b + d must be even
    # If b is odd, d must be odd; if b is even, d must be even
    
    # Let's iterate over small |d| values and check if there's a valid pair
    
    # For |d| = 1: a = (3b+1)/2 or (3b-1)/2
    #   If b is odd: 3b is odd, so 3b±1 is even, a is integer
    #   Check sigma((3b+1)//2) == sigma(b) and sigma((3b-1)//2) == sigma(b)
    
    # For |d| = 2: a = (3b+2)/2 or (3b-2)/2
    #   If b is even: 3b is even, so 3b±2 is even, a is integer
    #   Check similarly
    
    # This approach lets us check small d values efficiently!
    
    # Let's precompute for each d value whether we can find a pair
    
    # For d = 1: a = (3b+1)//2, need b odd, a <= MAX_N, a >= 1
    #   sigma(a) == sigma(b)?
    # For d = -1: a = (3b-1)//2, need b odd, a <= MAX_N, a >= 1
    # For d = 2: a = (3b+2)//2 = 3b//2 + 1, need b even
    # For d = -2: a = (3b-2)//2, need b even
    
    # Let's check d values in order of |d|: 1, -1, 2, -2, 3, -3, ...
    
    # For each d, compute candidate a values and check sigma equality
    
    # Vectorized approach for each d:
    
    for d in range(1, 100):  # Check small d values first
        if time.time() > deadline - 5:
            break
            
        # d and -d
        for dd in [d, -d]:
            if time.time() > deadline - 5:
                    break
            
            # a = (3b + dd) / 2 must be integer
            # 3b + dd must be even
            # If dd is odd: b must be odd
            # If dd is even: b must be even
            
            if dd % 2 == 1:
                # b must be odd
                b_vals = np.arange(1, MAX_N + 1, 2)  # odd b values
            else:
                # b must be even
                b_vals = np.arange(2, MAX_N + 1, 2)  # even b values
            
            # a = (3b + dd) // 2
            a_vals = (3 * b_vals + dd) // 2
            
            # Filter valid a: 1 <= a <= MAX_N, a != b
            mask = (a_vals >= 1) & (a_vals <= MAX_N) & (a_vals != b_vals)
            if not np.any(mask):
                continue
            
            a_valid = a_vals[mask]
            b_valid = b_vals[mask]
            
            # Check sigma(a) == sigma(b)
            sigma_a = sigma[a_valid]
            sigma_b = sigma[b_valid]
            
            match_mask = sigma_a == sigma_b
            if np.any(match_mask):
                # Found matches with this d value
                # The error is |dd| / (2b), so we want the largest b
                # to minimize the error
                matched_b = b_valid[match_mask]
                matched_a = a_valid[match_mask]
                
                # Find the pair with largest b (minimizes |dd|/(2b))
                max_b_idx = np.argmax(matched_b)
                a_candidate = int(matched_a[max_b_idx])
                b_candidate = int(matched_b[max_b_idx])
                
                # Compare with best
                num = abs(dd)
                den = 2 * b_candidate
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_pair = (a_candidate, b_candidate)
                    
                    # If we found d=1 with a large b, that's likely near-optimal
                    # We can break early if d=1 and b is large
                    if d == 1:
                        # Keep searching for even larger b with d=1
                        pass
    
    a, b = best_pair
    return [int(a), int(b)]
