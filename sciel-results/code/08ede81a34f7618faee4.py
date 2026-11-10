import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    limit = MAX_N + 1
    
    # Build smallest prime factor sieve up to MAX_N + 1
    # Using a list is faster for small N in Python, but let's try to be efficient.
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number in [1, MAX_N+1]
    pp = [1] * (limit + 1)
    
    for n in range(2, limit + 1):
        x = n
        val = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= p ** e
        pp[n] = val
        
    # Check deadline
    if time.time() > deadline:
        # Fallback: return a known good candidate or min
        return MIN_N

    # Use NumPy for faster maximization
    # pp[MIN_N : MAX_N + 2] gives pp[n] for n in [MIN_N, MAX_N+1]
    # We want max of pp[i] * pp[i+1] for i in [MIN_N, MAX_N]
    pp_slice = np.array(pp[MIN_N:MAX_N + 2], dtype=np.int64)
    products = pp_slice[:-1] * pp_slice[1:]
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = MIN_N + max_idx
    
    return best_n
