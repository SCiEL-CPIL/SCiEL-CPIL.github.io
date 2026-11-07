import numpy as np
import time
from fractions import Fraction

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    N = P['maximum']
    
    # Deadline guard
    start_time = time.time()
    timeout = 82.0  # 8s margin before 90s limit
    
    # 1. Compute sigma(n) for n in [1, N] using SPF sieve
    # Step 1a: Compute smallest prime factor (SPF)
    spf = np.zeros(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    
    if time.time() - start_time > timeout:
        return [11, 6]
    
    # Step 1b: Compute sigma(n) using SPF
    sigma = np.zeros(N + 1, dtype=np.int64)
    p_pow = np.zeros(N + 1, dtype=np.int64)
    sigma[1] = 1
    p_pow[1] = 1
    
    for n in range(2, N + 1):
        p = spf[n]
        m = n // p
        if m % p == 0:
            p_pow[n] = p_pow[m] * p
        else:
            p_pow[n] = p
        m2 = n // p_pow[n]
        if p == 2:
            sigma[n] = sigma[m2] * (p_pow[n] * 2 - 1)
        else:
            sigma[n] = sigma[m2] * (p_pow[n] * p - 1) // (p - 1)
    
    if time.time() - start_time > timeout:
        return [11, 6]
    
    # 2. Group indices by sigma value
    indices = np.arange(1, N + 1, dtype=np.int64)
    
    order = np.argsort(sigma[indices], kind='stable')
    sorted_indices = indices[order]
    sorted_sigmas = sigma[sorted_indices]
    
    if time.time() - start_time > timeout:
        return [11, 6]
    
    # 3. Find best pair using binary search within each block
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # This is equivalent to finding a, b in the same sigma-fiber
    # such that a/b is as close to 3/2 as possible
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    n_items = len(sorted_indices)
    i = 0
    while i < n_items:
        if time.time() - start_time > timeout:
            break
            
        current_sigma = sorted_sigmas[i]
        j = i + 1
        while j < n_items and sorted_sigmas[j] == current_sigma:
            j += 1
            
        block_size = j - i
        
        if block_size >= 2:
            block = sorted_indices[i:j]
            block_list = block.tolist()
            len_block = block_size
            
            # For each element as b, find a in block closest to 1.5 * b
            # Use binary search for efficiency
            for idx_b in range(len_block):
                b = block_list[idx_b]
                target = 1.5 * b
                
                pos = np.searchsorted(block, target)
                
                # Check neighbors around the insertion point
                for pos_check in range(max(0, pos - 2), min(len_block, pos + 3)):
                    a = int(block[pos_check])
                    if a == b:
                        continue
                    # Exact rational error: |a/b - 3/2| = |2a - 3b| / (2b)
                    err_num = abs(2 * a - 3 * b)
                    err_den = 2 * b
                    err = err_num / err_den
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        
        i = j
    
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
