import numpy as np
import time
import math

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    # 1. Sieve of Eratosthenes to find primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Precompute LPF (Largest Prime Factor) for all numbers using a sieve approach
    # Process primes in increasing order; each prime overwrites LPF for its multiples,
    # so the last prime to touch a number is its largest prime factor.
    lpf = np.ones(N + 1, dtype=np.int32)
    for p in primes:
        lpf[p::p] = p
    
    if time.time() > deadline:
        return (1, N - 1)
    
    # 3. Sort numbers by their LPF to enable incremental search
    numbers = np.arange(1, N + 1)
    lpf_vals = lpf[numbers]
    
    # Sort by lpf value
    sort_idx = np.argsort(lpf_vals, kind='stable')
    sorted_numbers = numbers[sort_idx]
    sorted_lpf = lpf_vals[sort_idx]
    
    # Iterate through unique LPF values in increasing order
    unique_lpf = np.unique(sorted_lpf)
    
    in_S = np.zeros(N + 1, dtype=bool)
    
    for M in unique_lpf:
        if time.time() > deadline:
            break
        
        # Add all numbers with lpf == M to S
        mask = (sorted_lpf == M)
        new_numbers = sorted_numbers[mask]
        
        for x in new_numbers:
            x = int(x)
            in_S[x] = True
            b = N - x
            if 1 <= b <= N and in_S[b]:
                return (x, b)
    
    # Fallback: brute force scan
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    cost = np.maximum(lpf[a], lpf[b])
    min_idx = np.argmin(cost)
    return (int(a[min_idx]), int(b[min_idx]))
