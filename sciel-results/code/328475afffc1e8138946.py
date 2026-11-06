"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
        # For each multiple of p, set lpf[multiple] = max(lpf[multiple], p)
        # Since we iterate primes in increasing order, the last assignment wins
        multiples = np.arange(p, limit + 1, p)
        lpf[multiples] = p
    
    # For n in [N_MIN, N_MAX], we want to minimize max(lpf[n], lpf[n+1])
    # Create arrays for n and n+1
    n_range = np.arange(N_MIN, N_MAX + 1)
    lpf_n = lpf[n_range]
    lpf_n1 = lpf[n_range + 1]
    
    # The largest prime factor of n(n+1) is max(lpf[n], lpf[n+1])
    max_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find the minimum value and corresponding n
    min_val = np.min(max_lpf)
    # Get all indices where this minimum occurs, pick the first
    min_indices = np.where(max_lpf == min_val)[0]
    best_n = n_range[min_indices[0]]
    
    return int(best_n)
# EVOLVE-BLOCK-END
