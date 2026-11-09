import numpy as np
import time
import math
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # Safety margin for 90s limit
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma using NumPy additive sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [6, 11]

    # 2. Group by sigma value using NumPy argsort for speed
    sigma_vals = sigma[1:]  # values for n=1..MAX_N
    order = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[order]
    sorted_indices = order + 1  # actual n values (1-indexed)
    
    # Find group boundaries
    change_mask = np.diff(sorted_sigma) != 0
    boundaries = np.where(change_mask)[0] + 1
    start_idx = np.concatenate(([0], boundaries))
    end_idx = np.concatenate((boundaries, [len(sorted_sigma)]))
    
    if time.time() > deadline - 5:
        return [6, 11]

    # 3. Find best pair: minimize |a/b - 3/2| subject to sigma(a)=sigma(b)
    best_err = float('inf')
    best_pair = (6, 11)  # Fallback: sigma(6)=12=sigma(11)
    
    num_groups = len(start_idx)
    
    for gi in range(num_groups):
        if time.time() > deadline - 2:
            break
            
        group_size = end_idx[gi] - start_idx[gi]
        
        if group_size < 2:
            continue
        
        # Extract sorted list of values in this bucket
        vals = sorted_indices[start_idx[gi]:end_idx[gi]]
        
        # For each b in vals, find a in vals closest to 1.5*b using binary search
        for bi in range(group_size):
            b = int(vals[bi])
            target = int(1.5 * b)
            
            # Binary search in vals for target
            idx = bisect.bisect_left(vals, target)
            
            # Check idx and idx-1
            if idx < group_size:
                a = int(vals[idx])
                if a != b:
                    diff = abs(2 * a - 3 * b)
                    if diff > 0:
                        err = diff / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
            if idx > 0:
                a = int(vals[idx-1])
                if a != b:
                    diff = abs(2 * a - 3 * b)
                    if diff > 0:
                        err = diff / (2.0 * b)
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
                    
    return [int(best_pair[0]), int(best_pair[1])]
