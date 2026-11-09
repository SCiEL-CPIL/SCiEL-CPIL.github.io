import numpy as np

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    
    # F[i] will store the 23-smooth part of i
    F = np.ones(max_val + 1, dtype=np.int64)
    
    # Sieve to compute 23-smooth parts
    for p in primes:
        for m in range(p, max_val + 1, p):
            x = m
            while x % p == 0:
                F[m] *= p
                x //= p
                
    # Sliding window distinct count
    # Window is F[n+1] ... F[n+24]
    # Start with n=1000
    # Window indices: 1001 to 1024
    
    # Initialize counts for the first window
    # Using a dictionary for counts
    counts = {}
    distinct = 0
    
    # Add first 24 elements
    for i in range(1001, 1025):
        val = F[i]
        if val in counts:
            counts[val] += 1
        else:
            counts[val] = 1
            distinct += 1
            
    min_distinct = distinct
    best_n = 1000
    
    # Slide the window
    # n goes from 1001 to 1000000
    # For a given n, window is F[n+1] ... F[n+24]
    # Previous window (n-1) was F[n] ... F[n+23]
    # So we remove F[n] and add F[n+24]
    
    for n in range(1001, max_n + 1):
        # Remove F[n]
        val_out = F[n]
        counts[val_out] -= 1
        if counts[val_out] == 0:
            distinct -= 1
            # Optional: del counts[val_out] to save memory, but dict lookup is O(1)
            
        # Add F[n+24]
        val_in = F[n+24]
        if val_in in counts:
            counts[val_in] += 1
        else:
            counts[val_in] = 1
            distinct += 1
            
        if distinct < min_distinct:
            min_distinct = distinct
            best_n = n
            
        # Early exit if we hit 1? Unlikely but possible.
        if min_distinct == 1:
            break

    return best_n
