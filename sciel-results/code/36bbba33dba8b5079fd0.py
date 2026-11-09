import numpy as np
import time

def construct():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MAX = 1000000
    MAX_VAL = N_MAX + 24
    
    # Precompute small part
    small_part = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= MAX_VAL:
            multiples = np.arange(power, MAX_VAL + 1, power)
            small_part[multiples] *= p
            power *= p
    
    best_n = 1000
    min_distinct = 100
    
    start_time = time.time()
    deadline = 80  # Safety margin
    
    sp = small_part
    
    for n in range(1000, N_MAX + 1):
        if (n & 4095) == 0 and time.time() - start_time > deadline:
            break
        
        window = sp[n+1 : n+25]
        # Use np.unique for counting distinct
        distinct_count = len(np.unique(window))
        
        if distinct_count < min_distinct:
            min_distinct = distinct_count
            best_n = n
            if min_distinct <= 2:
                # Try to find if we can do better with a focused search
                break
    
    return best_n

construct()
