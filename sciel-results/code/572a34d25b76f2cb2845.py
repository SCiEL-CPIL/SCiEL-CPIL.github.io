import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    min_n = 10000
    max_n = 1000000
    # We need lpf up to max_n + 1
    limit = max_n + 1
    N = limit + 1  # Size of array
    
    # Initialize array for largest prime factor
    lpf = np.zeros(N, dtype=np.int32)
    
    # Sieve for largest prime factors
    # Iterate i from 2 to limit
    # If lpf[i] is 0, i is prime
    # Mark multiples of i with i
    # Since we go in increasing order, the last mark is the largest prime factor
    for i in range(2, limit + 1):
        if lpf[i] == 0:
            # i is prime
            # Mark all multiples of i
            # j starts at i, goes to limit, step i
            for j in range(i, limit + 1, i):
                lpf[j] = i
                
    if time.time() > deadline:
        # Fallback
        return min_n

    # Compute M(n) = max(lpf[n], lpf[n+1]) for n in [min_n, max_n]
    # n ranges from min_n to max_n
    # lpf[n] corresponds to indices min_n ... max_n
    # lpf[n+1] corresponds to indices min_n+1 ... max_n+1
    
    # Extract slices
    # lpf has size N = max_n + 2. Indices 0 .. max_n+1.
    # We need lpf[min_n : max_n+1] and lpf[min_n+1 : max_n+2]
    
    left = lpf[min_n:max_n + 1]
    right = lpf[min_n + 1:max_n + 2]
    
    max_factors = np.maximum(left, right)
    min_idx = np.argmin(max_factors)
    
    best_n = min_n + min_idx
    
    return int(best_n)
