"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute small-prime part for each number up to max_val
    # small_part[m] = product of p^a for p in small_primes where p^a || m
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        p_power = p
        while p_power <= max_val:
            # Multiply small_part by p for all indices divisible by p_power
            idx = np.arange(p_power, max_val + 1, p_power)
            small_part[idx] *= p
            p_power *= p
            
    # Now small_part[m] is the complete factor of m composed of primes < 24
    
    # We want to minimize the number of distinct small_part values in the window
    # Window for n is small_part[n+1 : n+25]
    # n ranges from 1000 to 1000000
    
    # Use sliding window view
    # windows[i] = small_part[i : i+24]
    windows = np.lib.stride_tricks.sliding_window_view(small_part, 24)
    
    # We need windows starting at index 1001 (for n=1000) to 1000001 (for n=1000000)
    # windows[1001] corresponds to small_part[1001:10025]
    # windows[1000001] corresponds to small_part[1000001:1000025]
    
    # Extract relevant windows
    # Start index in windows array: 1001
    # End index in windows array: 1000001 (inclusive) -> slice 1000002
    relevant_windows = windows[1001 : 1000002]
    
    # Sort each window to count distinct values
    # Note: This creates a copy, which is memory intensive but fast
    sorted_windows = np.sort(relevant_windows, axis=1)
    
    # Count distinct values: 1 + number of adjacent differences
    # If a row is [1, 1, 2, 2], diffs are [0, 1, 0], sum=1, distinct=2
    distinct_counts = np.sum(sorted_windows[:, 1:] != sorted_windows[:, :-1], axis=1) + 1
    
    # Find the index with the minimum distinct count
    min_idx = np.argmin(distinct_counts)
    
    # Map back to n
    # min_idx is 0-based index in relevant_windows
    # relevant_windows[0] corresponds to n=1000
    best_n = 1000 + min_idx
    
    return best_n

# EVOLVE-BLOCK-END
