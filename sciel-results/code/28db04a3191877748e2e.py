import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # The task is to find n in [10000, 1000000] minimizing max(LPF(n), LPF(n+1))
    # We use a sieve to compute LPF for all numbers up to 1000001.
    
    max_val = 1000001
    size = max_val + 1
    
    # Sieve of Eratosthenes
    is_prime = np.ones(size, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    limit = int(np.sqrt(max_val)) + 1
    for i in range(2, limit):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return 10080
    
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor
    lpf = np.zeros(size, dtype=np.int64)
    
    # Update LPF for multiples of each prime
    # Primes are in increasing order, so the last assignment is the largest prime factor
    for p in primes:
        lpf[p::p] = p
        
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return 10080
    
    # Extract LPF for n and n+1 for n in [10000, 1000000]
    # n ranges from 10000 to 1000000
    # n+1 ranges from 100001 to 1000001
    
    lpf_n = lpf[10000:1000001]
    lpf_n1 = lpf[10001:1000002]
    
    # Find n that minimizes the maximum LPF
    max_lpf_pairs = np.maximum(lpf_n, lpf_n1)
    min_idx = np.argmin(max_lpf_pairs)
    best_n = 10000 + min_idx
    
    return int(best_n)
