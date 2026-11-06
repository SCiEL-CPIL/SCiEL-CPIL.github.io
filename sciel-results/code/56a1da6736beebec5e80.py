import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin for the 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    MAX_N = 1000000
    MIN_N = 1000
    WINDOW_SIZE = 24
    MAX_IDX = MAX_N + WINDOW_SIZE  # We need smooth parts up to n+24
    
    # Precompute smooth parts
    smooth = np.ones(MAX_IDX + 1, dtype=np.int64)
    
    for p in primes:
        # Compute exponents for this prime
        exp = np.zeros(MAX_IDX + 1, dtype=np.int8)
        power = p
        while power <= MAX_IDX:
            # Add 1 to all multiples of 'power'
            exp[power::power] += 1
            power *= p
        
        # Create lookup table for p^k
        max_exp = int(exp.max())
        lookup = np.array([p**k for k in range(max_exp + 1)], dtype=np.int64)
        
        # Apply exponents
        smooth *= lookup[exp]
    
    # Sliding window to find minimum distinct count
    # We need to find n in [MIN_N, MAX_N] such that 
    # len(set(smooth[n+1:n+25])) is minimized
    
    # Initialize frequency count for the first window
    window = smooth[MIN_N+1 : MIN_N+1+WINDOW_SIZE]
    freq = {}
    for val in window:
        freq[val] = freq.get(val, 0) + 1
    
    min_distinct = len(freq)
    best_n = MIN_N
    
    if min_distinct == 1:
        return best_n
    
    # Slide the window
    for n in range(MIN_N + 1, MAX_N + 1):
        if time.time() > deadline:
            break
        
        # Remove smooth[n]
        val_out = smooth[n]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
        
        # Add smooth[n+24]
        val_in = smooth[n + WINDOW_SIZE]
        freq[val_in] = freq.get(val_in, 0) + 1
        
        current_distinct = len(freq)
        if current_distinct < min_distinct:
            min_distinct = current_distinct
            best_n = n
            if min_distinct == 1:
                break
    
    return best_n
