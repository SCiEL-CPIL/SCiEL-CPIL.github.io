import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds safety margin
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF) using NumPy
    spf = np.arange(N + 2, dtype=np.int64)
    limit = int(math.isqrt(N)) + 1
    for i in range(2, limit):
        if spf[i] == i:
            # Vectorized update for multiples of i
            # We only want to update if the current spf value is the number itself (i.e., not yet assigned)
            start = i * i
            multiples = np.arange(start, N + 2, i, dtype=np.int64)
            # Check which of these multiples haven't been assigned a smaller prime factor
            # Since we iterate i in increasing order, spf[multiples] == multiples means it's still prime (unassigned)
            mask = spf[multiples] == multiples
            if np.any(mask):
                spf[multiples[mask]] = i

    # Check time
    if time.time() > deadline:
        return MIN_N # Fallback

    # 2. Compute powerful part for all numbers up to N
    # Using a Python list for powerful is often faster for this sequential dependency in a Python loop
    # than a NumPy array due to lower overhead per access in the tight loop
    powerful = [1] * (N + 2)
    
    for k in range(2, N + 2):
        p = int(spf[k])
        m = k
        e = 0
        while m % p == 0:
            m //= p
            e += 1
        
        if e >= 2:
            powerful[k] = powerful[m] * (p ** e)
        else:
            powerful[k] = powerful[m]
            
    # Check time
    if time.time() > deadline:
        return MIN_N # Fallback

    # 3. Vectorized scan for max product
    # Convert powerful list to NumPy array for vectorized operations
    powerful_np = np.array(powerful, dtype=np.int64)
    
    # n ranges from MIN_N to N-1
    # We need powerful[n] and powerful[n+1]
    # Slicing is efficient
    p1 = powerful_np[MIN_N:N]
    p2 = powerful_np[MIN_N+1:N+1]
    
    products = p1 * p2
    best_idx = int(np.argmax(products))
    best_n = MIN_N + best_idx
    
    return best_n
