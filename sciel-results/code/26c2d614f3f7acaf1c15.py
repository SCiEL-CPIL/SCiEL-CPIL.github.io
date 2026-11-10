import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 50.0 # seconds
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range for n: 1000 to 1000000
    # Max value needed: 1000000 + 24
    max_val = 1000024
    
    # Precompute small-prime parts for 1 to max_val
    # small_parts[i] is the part of i composed of primes < 24
    small_parts = np.ones(max_val + 1, dtype=np.int32)
    
    # For each prime, we need to extract its power from each number
    # This is effectively: for each x, x = small_part * large_part
    # small_part = product of p^v_p(x) for p in primes
    
    # We can do this by dividing x by p repeatedly.
    # Using NumPy arrays for speed
    
    x = np.arange(1, max_val + 1, dtype=np.int64)
    small_parts[1:] = x.copy()
    
    for p in primes:
        # We need to divide small_parts by p as many times as possible
        # mask = small_parts % p == 0
        # while any(mask):
        #     small_parts[mask] //= p
        #     mask = small_parts % p == 0
        
        # Optimization:
        # max power of p is p^k <= max_val
        # k = floor(log_p(max_val))
        # We can iterate k from 1 to k_max
        # But we need to know for each x how many times p divides it.
        
        # Alternative:
        # For each multiple of p, divide.
        # For each multiple of p^2, divide again.
        # ...
        # This is equivalent to:
        # For k in range(1, max_k+1):
        #     mask = small_parts % (p**k) == 0  # This is wrong, we need to divide if divisible by p
        #     Actually, if we divide by p once for all multiples, then check again.
        
        # Let's just do the while loop with vectorized operations.
        # It's usually fast enough for 10^6 elements and small number of primes.
        
        mask = small_parts % p == 0
        while np.any(mask):
            small_parts[mask] //= p
            mask = small_parts % p == 0
            
    # Now small_parts[i] is the extracted factor for i.
    
    # Find n in [1000, 1000000] minimizing distinct count in small_parts[n+1 : n+25]
    # Note: task says n+i for 1<=i<=24. So indices n+1 to n+24.
    
    best_n = 1000
    best_count = 100
    
    # Iterate through valid n
    # To speed up, we can check time periodically
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
            
        # Extract window
        # small_parts is 1-indexed effectively (0 is unused or 1)
        # We need small_parts[n+1] to small_parts[n+24]
        window = small_parts[n+1:n+25]
        
        # Count distinct
        # Using np.unique is robust
        distinct_count = len(np.unique(window))
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            
    return best_n
