import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin (90s kill limit)
    
    N = 1000000
    limit = N + 1
    
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    # Generate primes up to sqrt(limit) using simple sieve
    sqrt_limit = int(np.sqrt(limit)) + 1
    small_sieve = np.ones(sqrt_limit + 1, dtype=bool)
    small_sieve[0:2] = False
    for i in range(2, int(np.sqrt(sqrt_limit)) + 1):
        if small_sieve[i]:
            small_sieve[i*i::i] = False
    small_primes = np.where(small_sieve)[0]
    
    # Sieve: for each prime p <= sqrt(limit), mark multiples
    sieve_complete = True
    for p in small_primes:
        if time.time() > deadline:
            sieve_complete = False
            break
        lpf[p::p] = p
    
    # Positions where lpf==0 and index>=2 are primes > sqrt(limit)
    zero_mask = (lpf == 0)
    zero_mask[0:2] = False
    prime_indices = np.where(zero_mask)[0]
    lpf[prime_indices] = prime_indices
    
    if not sieve_complete:
        # Fallback: linear scan with partial sieve
        best_val = float('inf')
        best_n = 10000
        for n in range(10000, N + 1):
            a = int(lpf[n])
            b = int(lpf[n+1])
            if a == 0 or b == 0:
                continue
            val = a if a > b else b
            if val < best_val:
                best_val = val
                best_n = n
        return best_n
    
    # Vectorized scan for minimum max prime factor
    vals = np.maximum(lpf[10000:N+1], lpf[10001:N+2])
    min_idx = int(np.argmin(vals))
    best_n = 10000 + min_idx
    
    return best_n
