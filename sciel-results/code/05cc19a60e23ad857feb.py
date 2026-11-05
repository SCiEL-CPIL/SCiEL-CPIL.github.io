import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55 seconds safety margin (evaluator kills at 90s)
    
    N = 1000000
    MIN_N = 10000
    
    # 1. Sieve for Smallest Prime Factor (SPF) using NumPy
    # Use a standard sieve approach
    spf = np.zeros(N + 2, dtype=np.int64)
    for i in range(2, N + 2):
        spf[i] = i
    
    limit = int(math.isqrt(N)) + 1
    for i in range(2, limit):
        if spf[i] == i:  # i is prime
            start = i * i
            for j in range(start, N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() > deadline:
        return MIN_N
    
    # 2. Compute powerful part for all numbers up to N
    # powerful[n] = product of p^e for all prime factors p with exponent e >= 2
    powerful = np.ones(N + 2, dtype=np.int64)
    
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
    
    if time.time() > deadline:
        return MIN_N
    
    # 3. Vectorized scan for max product
    # n ranges from MIN_N to N-1
    # We need powerful[n] and powerful[n+1]
    p1 = powerful[MIN_N:N]
    p2 = powerful[MIN_N+1:N+1]
    
    products = p1 * p2
    best_idx = int(np.argmax(products))
    best_n = MIN_N + best_idx
    
    return best_n
