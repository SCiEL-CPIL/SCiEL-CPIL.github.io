import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000

    # Compute LPF (largest prime factor) for all numbers up to N_MAX+1
    # using a vectorized sieve approach
    size = N_MAX + 2
    lpf = np.zeros(size, dtype=np.int64)

    # Sieve: for each prime p, set lpf for multiples of p
    # We iterate through potential primes and mark their multiples
    for p in range(2, size):
        if lpf[p] == 0:  # p is prime
            # Mark all multiples of p: lpf[multiple] = p (overwrites with larger primes later)
            # Vectorized: set lpf[p::p] = p
            lpf[p::p] = p
        if time.time() > deadline:
            break

    # Compute objective: max(lpf(n), lpf(n+1)) for n in [N_MIN, N_MAX]
    lpf_n = lpf[N_MIN:N_MAX + 1]
    lpf_np1 = lpf[N_MIN + 1:N_MAX + 2]
    obj = np.maximum(lpf_n, lpf_np1)

    # Find the minimum
    min_val = np.min(obj)
    # Find all indices where obj == min_val, return the smallest n
    min_indices = np.where(obj == min_val)[0]
    best_n = int(N_MIN + min_indices[0])

    return best_n
