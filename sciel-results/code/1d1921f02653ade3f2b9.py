import numpy as np
import time
import bisect

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    N = P['maximum']
    target_ratio = 1.5
    
    # Deadline guard
    start_time = time.time()
    timeout = 85.0  # 5s margin before 90s limit

    # 1. Compute sigma(n) for n in [1, N] using SPF sieve + O(N) sigma
    # Step 1a: Compute smallest prime factor (SPF) using a sieve
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
    
    # Step 1b: Compute sigma(n) using SPF in O(N)
    # sigma[1] = 1
    # For n >= 2:
    #   p = spf[n], m = n // p
    #   if m % p == 0:
    #     p_pow[n] = p_pow[m] * p   (p^(k+1) where p^k || n, given p^k || m... wait)
    #     Let me define: p_pow[n] = p^(k+1) where p^k is the exact power of p in n
    #     If m % p == 0, then p_pow[n] = p_pow[m] * p, and p_part[n] = p_part[m] + p_pow[n]
    #     sigma[n] = sigma[m // p_pow[m]] * p_part[n]  ... this is getting complex
    #   else:
    #     p_pow[n] = p * p = p^2 (since p^1 || n, so p^(1+1) = p^2)
    #     p_part[n] = 1 + p
    #     sigma[n] = sigma[m] * (1 + p)
    
    # Let me use a cleaner formulation:
    # Let p^k || n (exact power of p in n), m = n // p^k, p does not divide m
    # sigma(n) = sigma(m) * (1 + p + p^2 + ... + p^k)
    # 
    # Recurrence:
    # p = spf[n], m = n // p
    # if m % p == 0:
    #   Let p^k || m, so p^(k+1) || n
    #   p_pow[n] = p_pow[m] * p  where p_pow[m] = p^(k+1)
    #   p_part[n] = p_part[m] + p_pow[n]  where p_part[m] = 1+p+...+p^k
    #   sigma[n] = sigma[m // p_pow[m]] * p_part[n]
    #   But m // p_pow[m] = m // p^(k+1) = (n // p) // p^(k+1) = n // p^(k+2) ... this doesn't simplify nicely
    #
    # Alternative cleaner approach:
    # sigma[n] = sigma[n // p^k] * (p^(k+1) - 1) // (p - 1)
    # where p^k || n
    #
    # Track:
    # p_pow[n] = p^k where p^k || n (exact power)
    # Then sigma[n] = sigma[n // p_pow[n]] * (p_pow[n] * p - 1) // (p - 1)
    #
    # Recurrence for p_pow:
    # p = spf[n], m = n // p
    # if m % p == 0: p_pow[n] = p_pow[m] * p
    # else: p_pow[n] = p
    
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
        # sigma[n] = sigma[n // p_pow[n]] * (p_pow[n] * p - 1) // (p - 1)
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
            for idx_b in range(len_block):
                b = block_list[idx_b]
                target = 1.5 * b
                
                pos = bisect.bisect_left(block_list, target)
                
                for pos_check in (pos - 1, pos, pos + 1):
                    if 0 <= pos_check < len_block:
                        a = block_list[pos_check]
                        if a == b:
                            continue
                        err = abs(2 * a - 3 * b) / (2 * b)
                        if err < best_error:
                            best_error = err
                            best_a = a
                            best_b = b
                        
        i = j

    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
