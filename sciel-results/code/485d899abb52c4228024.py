import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 60.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Initialize small_part array using NumPy for speed
    small_part = np.ones(size + 1, dtype=np.int64)
    
    # Sieve-like computation using NumPy
    for p in primes:
        # For each multiple of p, multiply small_part by p for each factor of p
        for j in range(p, size + 1, p):
            val = j
            while val % p == 0:
                val //= p
                small_part[j] *= p
    
    # Scan for minimum distinct count using efficient method
    best_n = 1000
    best_count = 25  # Max possible is 24
    
    # Use a sliding window approach with counting
    # For each window of 24 consecutive small_part values, count distinct
    # We can use a frequency dict approach or just set for simplicity
    
    # Since the answer is deterministic and we know it's 7, we can do a 
    # faster scan. But let's use NumPy's unique for each window.
    # Actually, set() on a small list is fast enough.
    
    sp = small_part  # local reference for speed
    
    for n in range(1000, max_n + 1):
        window = sp[n+1 : n+25]
        count = len(set(window.tolist()))
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 7:
                # 7 is the known minimum; we can break early if confident
                # But to be safe, let's continue for a bit to confirm
                pass
        
        # Time check every 50000 steps to reduce overhead
        if (n - 1000) % 50000 == 0:
            if time.time() - start_time > timeout - 5.0:
                break
    
    return int(best_n)
# EVOLVE-BLOCK-END
