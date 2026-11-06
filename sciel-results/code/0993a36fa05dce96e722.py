import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1
    
    # Sieve for primes
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    # Compute LPF (largest prime factor) for each number
    # lpf[k] will hold the largest prime factor of k
    lpf = np.zeros(limit + 1, dtype=np.int64)
    
    # Iterate through primes in ascending order.
    # For each prime p, mark all its multiples.
    # Since we go from small to large, the last prime to mark a number k
    # will be the largest prime factor of k.
    for p in primes:
        # Optimization: only mark multiples if p is within the range of interest?
        # No, we need lpf for all numbers to check n and n+1.
        # But we can stop early if p > limit? No, p is from primes list which is <= limit.
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
        
    if time.time() > deadline - 5:
        # Fallback: return a known good candidate or start of range
        # 10000 is valid
        return 10000
        
    # Scan all candidates
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
