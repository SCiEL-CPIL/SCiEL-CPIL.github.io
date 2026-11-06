"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def _build_lpf_sieve(n):
    """Build array where lpf[i] = largest prime factor of i, for i in 0..n."""
    lpf = np.zeros(n + 1, dtype=np.int32)
    lpf[0] = 0
    if n >= 1:
        lpf[1] = 1
    for i in range(2, n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, n + 1, i):
                lpf[j] = i
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    target = 123457
    
    # Precompute largest prime factor for all numbers up to target
    lpf = _build_lpf_sieve(target)
    
    # Vectorized search: for a in [1, target//2], b = target - a
    # Compute max(P(a), P(b)) for all splits at once
    half = target // 2
    a_vals = np.arange(1, half + 1, dtype=np.int32)
    b_vals = target - a_vals
    
    pa = lpf[a_vals]
    pb = lpf[b_vals]
    
    # Element-wise max of the two largest prime factors
    max_pf = np.maximum(pa, pb)
    
    # Find the index with minimum max_pf
    min_idx = np.argmin(max_pf)
    best_max_pf = int(max_pf[min_idx])
    best_a = int(a_vals[min_idx])
    best_b = int(b_vals[min_idx])
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
