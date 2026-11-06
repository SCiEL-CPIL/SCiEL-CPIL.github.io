import math
import time
import numpy as np

DEADLINE = 85.0  # safety margin before 90s kill

def construct():
    start_time = time.time()
    N = 1000000
    MAX_VAL = N + 1
    
    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i, v in enumerate(is_prime) if v]
    
    # Initialize powerful part array
    P = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    # Compute powerful parts
    for p in primes:
        pe = p * p
        if pe > MAX_VAL:
            break
        pe_next = pe * p
        while pe <= MAX_VAL:
            indices = np.arange(pe, MAX_VAL + 1, pe)
            if pe_next > MAX_VAL:
                P[indices] *= pe
                break
            else:
                mask = indices % pe_next != 0
                P[indices[mask]] *= pe
                pe = pe_next
                pe_next = pe * p
    
    if time.time() - start_time > DEADLINE:
        return 10000
    
    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    products = P[10000:N+1] * P[10001:N+2]
    idx = int(np.argmax(products))
    best_n = 10000 + idx
    
    return best_n
