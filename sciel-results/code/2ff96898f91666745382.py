import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 3_000_000
    
    sigmas = np.zeros(N + 1, dtype=np.uint32)
    for i in range(1, N + 1):
        sigmas[i::i] += i
    
    indices = np.arange(1, N + 1, dtype=np.int64)
    sig_vals = sigmas[1:]
    
    order = np.argsort(sig_vals, kind='stable')
    sorted_sigmas = sig_vals[order]
    sorted_indices = indices[order]
    
    best_a = 11
    best_b = 6
    best_err = abs(11.0 / 6.0 - 1.5)
    
    i = 0
    n_items = N
    while i < n_items:
        if time.time() - start_time > deadline - 5:
            break
        j = i
        current_sigma = sorted_sigmas[i]
        j = np.searchsorted(sorted_sigmas, current_sigma, side='right', left=i)
        
        group_size = j - i
        if group_size >= 2:
            group_vals = sorted_indices[i:j].copy()
            group_vals.sort()
            
            for b_val in group_vals:
                target_a = 1.5 * b_val
                pos = bisect.bisect_left(group_vals, target_a)
                for p in (pos, pos - 1):
                    if 0 <= p < group_size:
                        a_val = group_vals[p]
                        err = abs(a_val / b_val - 1.5)
                        if err < best_err:
                            best_err = err
                            best_a = a_val
                            best_b = b_val
                            if best_err == 0.0:
                                return [int(best_a), int(best_b)]
        i = j
    
    return [int(best_a), int(best_b)]
