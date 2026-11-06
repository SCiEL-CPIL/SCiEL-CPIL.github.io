"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_n = 1000000
    max_val = max_n + 24  # need to compute up to n+24 where n <= 1000000
    
    # Precompute the 23-smooth part for all numbers from 1 to max_val
    # smooth_part[m] = largest divisor of m with all prime factors < 24
    smooth_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # For each multiple of p, multiply smooth_part by p as many times as it divides
        # Actually, we need the full p-adic part
        for multiple in range(p, max_val + 1, p):
            # Count how many times p divides multiple
            temp = multiple
            while temp % p == 0:
                smooth_part[multiple] *= p
                temp //= p
    
    # Now slide a window of 24 and count distinct smooth_part values
    best_n = 1000
    best_count = 25  # worst case
    
    # Use a set to track distinct values in the window
    # For efficiency, use a sliding window with a frequency counter
    window_size = 24
    
    # Initialize window [1, 24] -> corresponds to n=0, but we need n >= 1000
    # Let's just iterate n from 1000 to max_n
    # For each n, the window is smooth_part[n+1] to smooth_part[n+24]
    
    # To make this fast, precompute and then use a sliding approach
    # Actually, let's just do it with a set for each window - but that's O(24) per window
    # With ~1 million windows, that's 24 million set operations - should be OK in ~5-10s
    
    # Better: use a frequency array approach
    # But smooth_part values can be large, so let's use a dict
    
    freq = {}
    distinct = 0
    
    # Initialize window for n=1000: indices 1001 to 1024
    for i in range(1001, 1025):
        v = smooth_part[i]
        freq[v] = freq.get(v, 0) + 1
        if freq[v] == 1:
            distinct += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n + 1):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Remove smooth_part[n] (which was n+1-1 = n, i.e., the leftmost element leaving)
        # Window for n-1 was [n, n+23]
        # Window for n is [n+1, n+24]
        # So remove index n, add index n+24
        
        v_out = smooth_part[n]
        freq[v_out] -= 1
        if freq[v_out] == 0:
            distinct -= 1
            del freq[v_out]
        
        v_in = smooth_part[n + 24]
        freq[v_in] = freq.get(v_in, 0) + 1
        if freq[v_in] == 1:
            distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
