import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    N_LIMIT = MAX_N + 1

    # Deadline guard
    DEADLINE = 85.0 # Seconds

    # 1. Compute Smallest Prime Factor (SPF) using a standard sieve
    spf = np.arange(N_LIMIT + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part for each number
    # p_part[i] is the powerful part of i
    p_part = np.ones(N_LIMIT + 1, dtype=np.int64)
    
    # We can vectorize parts of this, but the dependency p_part[i] depends on p_part[x] 
    # where x < i means we must iterate i in increasing order.
    # However, the inner loop is small (number of prime factors).
    
    for i in range(2, N_LIMIT + 1):
        if spf[i] == i:
            # i is prime
            p_part[i] = 1
        else:
            p = spf[i]
            x = i
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part[i] = (p ** e) * p_part[x]
            else:
                p_part[i] = p_part[x]

    # 3. Search for max product
    # We only care about n in [MIN_N, MAX_N]
    # Product is P(n) * P(n+1)
    # Indices: P(MIN_N) ... P(MAX_N)
    # P(MIN_N+1) ... P(MAX_N+1)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return a safe default if we ran out of time (unlikely)
        return MIN_N

    products = p_part[MIN_N:MAX_N + 1] * p_part[MIN_N + 1:MAX_N + 2]
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = MIN_N + max_idx
    
    return int(best_n)
