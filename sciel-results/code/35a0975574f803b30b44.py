"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def construct():
    N = 1000000
    MIN_N = 10000
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Proper SPF sieve
    spf = list(range(N + 2))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    spf = np.array(spf, dtype=np.int64)
    
    # 2. Compute powerful part using SPF
    pp = np.ones(N + 2, dtype=np.int64)
    
    for n in range(4, N + 2):
        if time.time() > deadline:
            break
        x = n
        while x > 1:
            p = int(spf[x])
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp[n] *= (p ** e)
    
    # 3. Vectorized scan for max product
    pp_slice = pp[MIN_N:N+1]
    products = pp_slice[:-1] * pp_slice[1:]
    best_idx = int(np.argmax(products))
    best_n = MIN_N + best_idx

    return best_n

# EVOLVE-BLOCK-END
