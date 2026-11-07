import time
import math
import bisect

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    
    # Fast sigma computation using divisor-sum sieve
    sigma = [0] * (MAX_N + 1)
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    # Group indices by sigma value
    groups = {}
    for i in range(1, MAX_N + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() - start_time > DEADLINE:
        return [11, 6]

    best_error = float('inf')
    best_a = None
    best_b = None
    
    # Process each group
    # We want to minimize |a/b - 1.5|
    # a and b must be in the same group.
    
    for s, lst in groups.items():
        if time.time() - start_time > DEADLINE:
            break
            
        if len(lst) < 2:
            continue
            
        # lst is already sorted because we appended in increasing order of i
        
        # Use two pointers to find pairs where a/b ≈ 1.5
        ptr_a = 0
        n_lst = len(lst)
        
        for b in lst:
            # Move ptr_a to be close to 1.5 * b
            target_a = 1.5 * b
            while ptr_a < n_lst and lst[ptr_a] < target_a:
                ptr_a += 1
                
            # Check ptr_a and ptr_a - 1
            for idx in [ptr_a, ptr_a - 1]:
                if 0 <= idx < n_lst:
                    a = lst[idx]
                    if a == b:
                        continue
                    err_num = abs(2*a - 3*b)
                    if err_num == 0:
                        continue
                    err = err_num / (2.0 * b)
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
    if best_a is None:
        return [11, 6]
        
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
