import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    # Max value we need: 1000000 + 24 = 1000024
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24, p^k || m
    
    # Initialize
    extracted = [1] * (MAX_VAL + 1)
    
    # For each small prime, update extracted for its multiples
    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            # Calculate the p-part of m
            factor = 1
            while val % p == 0:
                factor *= p
                val //= p
            extracted[m] *= factor

    # Now search for n that minimizes distinct extracted factors
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct
    
    # Use a frequency array to count distinct values in sliding window
    # Map extracted values to indices
    # First, collect all unique extracted values in range [N_MIN+1, MAX_VAL]
    unique_vals = set()
    for i in range(N_MIN + 1, MAX_VAL + 1):
        unique_vals.add(extracted[i])
    
    # Map values to indices
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    num_unique = len(unique_vals)
    
    # Create indexed array
    indexed = [0] * (MAX_VAL + 1)
    for i in range(N_MIN + 1, MAX_VAL + 1):
        indexed[i] = val_to_idx[extracted[i]]
    
    # Sliding window with frequency count
    freq = [0] * num_unique
    distinct = 0
    
    # Initialize window for n = N_MIN (covers n+1 to n+WINDOW)
    for i in range(N_MIN + 1, N_MIN + WINDOW + 1):
        idx = indexed[i]
        freq[idx] += 1
        if freq[idx] == 1:
            distinct += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = N_MIN
    
    # Slide the window
    for n in range(N_MIN + 1, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Remove n
        out_idx = indexed[n]
        freq[out_idx] -= 1
        if freq[out_idx] == 0:
            distinct -= 1
        
        # Add n + WINDOW
        in_idx = indexed[n + WINDOW]
        freq[in_idx] += 1
        if freq[in_idx] == 1:
            distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
