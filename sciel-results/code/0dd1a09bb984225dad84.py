import numpy as np

def construct():
    N = 1000001
    # 1. Sieve of Eratosthenes to find primes up to sqrt(N)
    limit = int(np.sqrt(N)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to N
    # Initialize LPF array with 0
    lpf = np.zeros(N + 1, dtype=np.int64)
    
    # For each prime, update the LPF for all its multiples
    for p in primes:
        # Multiples of p: p, 2p, 3p, ...
        # We want to set lpf[m] = p for all m divisible by p.
        # Since we iterate p in increasing order, the last p that touches m is the largest prime factor.
        # However, simply overwriting works because if p divides m, p is a prime factor.
        # The largest prime factor will be the largest p in this set.
        # So iterating p in increasing order and assigning lpf[m] = p is correct.
        multiples = np.arange(p, N + 1, p)
        lpf[multiples] = p
        
    # Any number > 1 that has lpf[x] == 0 is prime itself
    # (Because if it were composite, it would have a prime factor <= sqrt(x) <= 1000, 
    #  which would have been processed and set lpf[x] > 0)
    # Note: lpf[0] and lpf[1] are 0, which is fine as we start from 10000.
    remaining_primes = np.where((lpf == 0) & (np.arange(N + 1) > 1))[0]
    lpf[remaining_primes] = remaining_primes
    
    # 3. Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    start = 10000
    end = 1000000
    
    # We need max(lpf[i], lpf[i+1]) for i in [start, end]
    # lpf[i] is in lpf[start:end+1]
    # lpf[i+1] is in lpf[start+1:end+2]
    
    lpf_n = lpf[start:end+1]
    lpf_n1 = lpf[start+1:end+2]
    
    # Compute the metric for each n
    metrics = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum metric
    min_idx = np.argmin(metrics)
    
    # The actual n is start + min_idx
    best_n = start + min_idx
    
    return int(best_n)
