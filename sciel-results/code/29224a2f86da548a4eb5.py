# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = t0 + 82.0  # 8s safety margin
    
    N = 10_000_000
    
    # Sieve for sigma(n)
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # For d from 1 to N, add d to all multiples of d
    # Vectorized in chunks to manage memory
    for d in range(1, N + 1):
        if time.time() > deadline - 10.0:  # leave 10s for search
            break
        sigma[d::d] += d
    
    # Build index: sort by sigma value
    # sigma[1:] has N elements
    order = np.argsort(sigma[1:], kind='stable')
    sorted_sigma = sigma[1:][order]
    sorted_n = order + 1  # 1-indexed n values
    
    # Find unique sigma values and their ranges
    unique_sigmas, start_indices, counts = np.unique(sorted_sigma, return_index=True, return_counts=True)
    
    best_err_num = None
    best_err_den = None
    best_a, best_b = 11, 6
    
    n_groups = len(unique_sigmas)
    
    # Phase 1: for each group, iterate over b, find a ≈ 3b/2
    for i in range(n_groups):
        if time.time() > deadline:
            break
        
        count = int(counts[i])
        if count < 2:
            continue
        
        start = int(start_indices[i])
        nums = sorted_n[start:start + count]
        
        # Limit checks per group
        max_checks = min(count, 200)
        if count <= max_checks:
            b_indices = np.arange(count)
        else:
            b_indices = np.linspace(0, count - 1, max_checks).astype(int)
        
        for bi in b_indices:
            b = int(nums[bi])
            target_a_val = (3 * b) // 2
            pos = np.searchsorted(nums, target_a_val)
            for offset in range(-4, 5):
                idx = pos + offset
                if idx < 0 or idx >= count:
                    continue
                a = int(nums[idx])
                if a == b:
                    continue
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    continue
                err_den = 2 * b
                if best_err_num is None:
                    best_err_num = err_num
                    best_err_den = err_den
                    best_a, best_b = a, b
                else:
                    lhs = err_num * best_err_den
                    rhs = best_err_num * err_den
                    if lhs < rhs:
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a, best_b = a, b
    
    # Phase 2: iterate over a, find b ≈ 2a/3
    for i in range(n_groups):
        if time.time() > deadline:
            break
        
        count = int(counts[i])
        if count < 2:
            continue
        
        start = int(start_indices[i])
        nums = sorted_n[start:start + count]
        
        max_checks = min(count, 200)
        if count <= max_checks:
            a_indices = np.arange(count)
        else:
            a_indices = np.linspace(0, count - 1, max_checks).astype(int)
        
        for ai in a_indices:
            a = int(nums[ai])
            target_b_val = (2 * a) // 3
            pos = np.searchsorted(nums, target_b_val)
            for offset in range(-4, 5):
                idx = pos + offset
                if idx < 0 or idx >= count:
                    continue
                b = int(nums[idx])
                if a == b:
                    continue
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    continue
                err_den = 2 * b
                if best_err_num is None:
                    best_err_num = err_num
                    best_err_den = err_den
                    best_a, best_b = a, b
                else:
                    lhs = err_num * best_err_den
                    rhs = best_err_num * err_den
                    if lhs < rhs:
                        best_err_num = err_num
                        best_err_den = err_den
                        best_a, best_b = a, b
    
    best_a = int(best_a)
    best_b = int(best_b)
    return [best_a, best_b]

# EVOLVE-BLOCK-END
