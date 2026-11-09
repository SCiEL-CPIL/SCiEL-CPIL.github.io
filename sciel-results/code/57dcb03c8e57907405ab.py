import time
import math
import numpy as np

def construct():
    start_time = time.time()
    MAX_N = 1000000
    MIN_N = 10000
    N_LIMIT = MAX_N + 1

    # 1. Compute Smallest Prime Factor (SPF) using a more efficient sieve
    spf = np.arange(N_LIMIT + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N_LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i * i, N_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part for each number using DP
    # p_part[i] = powerful part of i
    p_part = np.ones(N_LIMIT + 1, dtype=np.int64)
    
    for i in range(2, N_LIMIT + 1):
        if spf[i] == i:
            # i is prime, powerful part is 1 (exponent 1)
            p_part[i] = 1
        else:
            p = spf[i]
            # Count exponent of p in i
            x = i
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                p_part[i] = (p ** e) * p_part[x]
            else:
                p_part[i] = p_part[x]

    # 3. Search for max product using NumPy for speed
    # Compute products for all adjacent pairs
    products = p_part[MIN_N:MAX_N + 1] * p_part[MIN_N + 1:MAX_N + 2]
    
    # Find the index of the maximum product
    max_idx = np.argmax(products)
    best_n = MIN_N + max_idx
    
    # Verify we're within time limit
    if time.time() - start_time > 80:
        # Return best found so far (which is already computed)
        pass
    
    return int(best_n)
