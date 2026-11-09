"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range: n in [1000, 1000000], so n+i in [1001, 1000024]
    min_num = 1001
    max_num = 1000024
    total_nums = max_num - min_num + 1  # 999024 numbers
    
    # Precompute extracted factors (the "large part" after removing small primes)
    # For each number, extracted_factor = number / (product of all small-prime factors)
    # i.e., divide out all small prime factors completely
    
    # Vectorized approach: for each small prime, divide out its power
    # Start with array of all numbers
    nums = np.arange(min_num, max_num + 1, dtype=np.int64)
    extracted = nums.copy()
    
    for p in small_primes:
        # For each number, divide out all factors of p
        # We need to repeatedly divide by p while divisible
        # Vectorized: use modulo to check divisibility, then divide
        # This is tricky to fully vectorize for repeated division
        # Alternative: for each p, compute the p-part of each number
        
        # Method: for each number, while divisible by p, divide by p
        # Vectorized approach: process in chunks or use a loop for each p
        # Since there are only 9 primes, and each number has limited p-factors,
        # we can do this efficiently
        
        # For prime p, the p-part is p^(v_p(n))
        # We can compute this by repeated division
        # Vectorized: 
        for _ in range(20):  # enough iterations (2^20 > 1e6)
            mask = (extracted % p == 0)
            if not np.any(mask):
                break
            extracted[mask] = extracted[mask] // p
    
    # Now extracted[i] is the large part of (min_num + i)
    # For n, the window is [n+1, n+24], which corresponds to indices [n+1-min_num, n+24-min_num]
    # i.e., indices [n-1000, n+14] in the extracted array (0-indexed)
    # Wait: n+1 - min_num = n+1-1001 = n-1000
    # n+24 - min_num = n+24-1001 = n-1077... that's wrong
    
    # Let me recompute: 
    # extracted array index j corresponds to number (min_num + j) = 1001 + j
    # For number n+i, its index is (n+i) - 1001 = n + i - 1001
    # For i=1: index = n - 1000
    # For i=24: index = n + 24 - 1001 = n - 977
    # So window in extracted array: indices [n-1000, n-977], which is 24 elements
    
    # n ranges from 1000 to 1000000
    # For n=1000: indices [0, 23]
    # For n=1000000: indices [999000, 999023]
    # extracted has length 999024, so max index is 999023. Good.
    
    # Now slide window of size 24 over extracted array
    # For each window position, count distinct values
    
    # Use sliding window with frequency count
    window_size = 24
    n_values = 1000000 - 1000 + 1  # 999001 possible n values
    
    # Initialize frequency for first window
    from collections import defaultdict
    freq = defaultdict(int)
    distinct_count = 0
    
    # First window: extracted[0:24]
    for i in range(24):
        val = extracted[i]
        if freq[val] == 0:
            distinct_count += 1
        freq[val] += 1
    
    best_score = distinct_count
    best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Check time
        if time.time() > deadline - 5:
            break
        
        # Remove element at index n-1001 (the one leaving)
        out_idx = n - 1001
        out_val = extracted[out_idx]
        freq[out_val] -= 1
        if freq[out_val] == 0:
            distinct_count -= 1
            del freq[out_val]
        
        # Add element at index n-977 (the one entering)
        in_idx = n - 977
        in_val = extracted[in_idx]
        if freq[in_val] == 0:
            distinct_count += 1
        freq[in_val] += 1
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:  # Can't do better than 1
                break
    
    return best_n
# EVOLVE-BLOCK-END
