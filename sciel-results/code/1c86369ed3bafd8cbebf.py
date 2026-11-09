import time
import sys
import numpy as np
from collections import Counter

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N = 1000000
    MAX_VAL = N + 24
    
    # Precompute the small part for all numbers up to MAX_VAL using NumPy
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            indices = np.arange(power, MAX_VAL + 1, power)
            sp[indices] *= p
            power *= p
    
    # Now sp[i] contains the complete factor of i composed of primes strictly below 24.
    
    best_n = 1000
    start_time = time.time()
    deadline = 85
    
    # Sliding window approach with frequency count
    # Window is sp[n+1], sp[n+2], ..., sp[n+24] for n in [1000, 1000000]
    
    # Initialize frequency dict for the first window: sp[1001] to sp[1024]
    initial_window = sp[1001:1025]
    freq = Counter(initial_window.tolist())
    distinct_count = len(freq)
    
    min_distinct = distinct_count
    best_n = 1000
    
    if min_distinct == 1:
        return best_n
    
    # Slide the window
    for n in range(1001, N + 1):
        if (n & 8191) == 0:
            if time.time() - start_time > deadline:
                break
        
        # Remove sp[n]
        val_out = int(sp[n])
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
            distinct_count -= 1
        
        # Add sp[n+24]
        val_in = int(sp[n + 24])
        if val_in not in freq:
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            
            if min_distinct == 1:
                break
    
    return best_n

construct()
