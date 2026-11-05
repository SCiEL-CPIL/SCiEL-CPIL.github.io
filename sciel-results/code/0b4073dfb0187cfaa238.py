import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW
    
    # Precompute extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in small_primes:
        multiples = np.arange(p, MAX_VAL + 1, p)
        p_parts = np.ones(len(multiples), dtype=np.int64)
        temp = multiples.copy()
        while np.any(temp % p == 0):
            mask = temp % p == 0
            temp[mask] //= p
            p_parts[mask] *= p
        extracted[multiples] *= p_parts
    
    # Convert to list for faster set operations in the search loop
    extracted_list = extracted.tolist()
    
    best_n = N_MIN
    best_count = WINDOW + 1  # worst case is 24 distinct
    
    # Search for n that minimizes distinct extracted factors in window [n+1, n+24]
    for n in range(N_MIN, N_MAX + 1):
        if time.time() - start_time > DEADLINE:
            break
        
        # Count distinct values in the window
        factors = set()
        for i in range(1, WINDOW + 1):
            factors.add(extracted_list[n + i])
        
        count = len(factors)
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n
