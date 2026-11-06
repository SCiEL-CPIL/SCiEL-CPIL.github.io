"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin: evaluator kills at 90s
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24
    
    # Precompute smooth parts
    # smooth_part[m] = factor of m composed of primes < 24
    smooth_part = [1] * (max_val + 1)
    
    for p in primes:
        for multiple in range(p, max_val + 1, p):
            m = multiple
            while m % p == 0:
                smooth_part[multiple] *= p
                m //= p
    
    best_n = min_n
    best_score = 25  # Max possible distinct is 24, but 25 is safe init
    
    # Sliding window approach with frequency counter
    # Window covers indices [n+1, n+24] in smooth_part
    # For n = min_n, window is smooth_part[min_n+1 : min_n+25]
    
    # Initialize frequency counter for first window
    window_vals = smooth_part[min_n+1 : min_n+25]
    freq = Counter(window_vals)
    distinct = len(freq)
    
    if distinct < best_score:
        best_score = distinct
        best_n = min_n
    
    # Slide the window
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline:
            break
        
        # Remove smooth_part[n] (leftmost element leaving window)
        # Previous window: [n, n+23] -> new window: [n+1, n+24]
        # Wait, let me re-index:
        # For candidate n, window is smooth_part[n+1 : n+25]
        # For candidate n-1, window is smooth_part[n : n+24]
        # So going from n-1 to n: remove smooth_part[n], add smooth_part[n+24]
        
        old_val = smooth_part[n]
        new_val = smooth_part[n+24]
        
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct -= 1
        
        freq[new_val] = freq.get(new_val, 0) + 1
        if freq[new_val] == 1:
            distinct += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score <= 2:
                # Extremely unlikely, but break early if found
                break
    
    return best_n
# EVOLVE-BLOCK-END
