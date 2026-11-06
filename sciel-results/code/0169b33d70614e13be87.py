"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import array
import numpy as np

def construct():
    # Explicit deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s margin
    
    target = 123457
    max_n = target - 1  # 123456
    
    # Use array of unsigned ints for LPF storage
    lpf = array.array('I', [0]) * (max_n + 1)
    if max_n >= 1:
        lpf[1] = 1
        
    # Sieve to find largest prime factor
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
                
    # Vectorized search using NumPy for speed and robustness
    lpf_np = np.array(lpf, dtype=np.uint32)
    
    half_target = target // 2
    a_vals = np.arange(1, half_target + 1, dtype=np.uint32)
    b_vals = target - a_vals
    
    pa = lpf_np[a_vals]
    pb = lpf_np[b_vals]
    
    cur_max = np.maximum(pa, pb)
    
    min_val = np.min(cur_max)
    idx = np.argmin(cur_max)
    
    best_a = int(a_vals[idx])
    best_b = int(b_vals[idx])
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
