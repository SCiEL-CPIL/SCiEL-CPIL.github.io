# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1 = 1000001
    
    # Sieve of Eratosthenes to find all primes up to limit
    is_prime = np.ones(limit + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = False
    
    primes = np.where(is_prime)[0]
    
    # Compute Largest Prime Factor (LPF) for each number up to limit
    # lpf[x] will store the largest prime factor of x
    lpf = np.zeros(limit + 1, dtype=np.int64)
    for p in primes:
        # For each multiple of p, set lpf[multiple] = p
        # Since we iterate primes in increasing order, the last assignment gives the largest prime factor
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    # Check deadline
    if time.time() > deadline - 5:
        # Fallback: just return a reasonable value
        return 10000
    
    # Scan all n in [N_MIN, N_MAX] to find min of max(lpf[n], lpf[n+1])
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    scores = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    best_score = int(scores[best_idx])
    
    return best_n
# EVOLVE-BLOCK-END
