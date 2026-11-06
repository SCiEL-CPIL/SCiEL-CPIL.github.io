"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 85  # 85s to be safe under 90s kill
    target = 123457
    max_n = target - 1  # 123456
    
    # Vectorized sieve to compute largest prime factor
    lpf = np.zeros(max_n + 1, dtype=np.int32)
    lpf[1] = 1
    
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            # Mark all multiples of i
            indices = np.arange(i, max_n + 1, i)
            # For each multiple, lpf should be max(current, i)
            # Since we iterate i in increasing order, i is the largest prime factor seen so far for each multiple
            # But we need the LARGEST prime factor, so we update only if i > current lpf
            current = lpf[indices]
            mask = (current == 0) | (i > current)
            lpf[indices[mask]] = i
    
    # Vectorized search: compute max(lpf[a], lpf[target-a]) for all a in [1, target-1]
    a_arr = np.arange(1, target)
    b_arr = target - a_arr
    pa = lpf[a_arr]
    pb = lpf[b_arr]
    cur_max = np.maximum(pa, pb)
    
    best_idx = np.argmin(cur_max)
    best_max = int(cur_max[best_idx])
    best_a = int(a_arr[best_idx])
    best_b = int(b_arr[best_idx])
    
    # Check if we achieved the theoretical minimum of 3
    # Since target is odd, one of a,b is even (P>=2) and one is odd
    # The odd one must be > 1 (unless it's 1, but P(1)=1)
    # If best_max == 3, that's optimal
    
    if time.time() > deadline - 5:
        # Safety: return what we have
        return [best_a, best_b]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
