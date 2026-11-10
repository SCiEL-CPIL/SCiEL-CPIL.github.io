import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Initialize small_part array using NumPy for speed
    small_part = np.ones(size + 1, dtype=np.int64)
    
    # Sieve-like computation using NumPy
    for p in primes:
        for j in range(p, size + 1, p):
            val = j
            while val % p == 0:
                val //= p
                small_part[j] *= p
    
    # Convert to Python list for faster set operations in the scan loop
    sp = small_part.tolist()
    
    best_n = 1000
    best_count = 25  # Max possible is 24, so 25 is infinity for count
    
    # Scan for minimum distinct count
    for n in range(1000, max_n + 1):
        window = sp[n+1 : n+25]
        count = len(set(window))
        if count < best_count:
            best_count = count
            best_n = n
            if best_count <= 2:
                break
        
        # Time check every 20000 steps to reduce overhead
        if (n - 1000) % 20000 == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
