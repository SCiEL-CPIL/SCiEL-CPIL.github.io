import sys
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 90.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Initialize small_part array using a list of integers
    small_part = [1] * (size + 1)
    
    # Sieve-like computation
    for p in primes:
        for j in range(p, size + 1, p):
            val = j
            while val % p == 0:
                val //= p
                small_part[j] *= p
    
    best_n = 1000
    best_count = 25  # Max possible is 24, so 25 is infinity for count
    
    # Scan for minimum distinct count
    # We can stop early if we find count 2 (since 1 is impossible)
    
    for n in range(1000, max_n + 1):
        # Extract slice and count distinct values
        window = small_part[n+1 : n+25]
        count = len(set(window))
        if count < best_count:
            best_count = count
            best_n = n
            if best_count == 2:
                break
        
        # Time check every 10000 steps
        if (n - 1000) % 10000 == 0:
            if time.time() - start_time > timeout - 5.0:
                break
                
    return best_n
# EVOLVE-BLOCK-END
