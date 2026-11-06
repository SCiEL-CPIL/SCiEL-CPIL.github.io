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
    
    best_max_pf = float('inf')
    best_a = 1
    best_b = target - 1
    
    # Iterate through all a from 1 to target//2
    # Using numpy for vectorized operations would be ideal, but let's do a simple loop
    # with the precomputed sieve for O(1) lookup
    
    for a in range(1, target // 2 + 1):
        if time.time() > deadline:
            break
        
        b = target - a
        
        pa = lpf[a]
        if pa >= best_max_pf:
            continue
            
        pb = lpf[b]
        if pb >= best_max_pf:
            continue
            
        current_max = max(pa, pb)
        if current_max < best_max_pf:
            best_max_pf = current_max
            best_a = a
            best_b = b

    return [best_a, best_b]
# EVOLVE-BLOCK-END
