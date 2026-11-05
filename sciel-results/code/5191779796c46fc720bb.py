import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 50  # safety margin before 60s limit
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000024  # 1000000 + 24
    
    # Precompute small-prime-part for all numbers 0..max_val
    small_part = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        power = p
        while power <= max_val:
            small_part[power::power] *= p
            power *= p
    
    # Convert to list for faster set operations in Python
    sp_list = small_part.tolist()
    
    best_n = 1000
    best_count = float('inf')
    
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        window = sp_list[n+1:n+25]
        cnt = len(set(window))
        if cnt < best_count:
            best_count = cnt
            best_n = n
            if best_count <= 1:
                break
    
    return best_n
