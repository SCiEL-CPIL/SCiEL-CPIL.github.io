import numpy as np
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = 10000000
    
    # Step 1: Sieve of Eratosthenes to get smallest prime factor (SPF)
    spf = np.arange(N + 1, dtype=np.int64)
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() > deadline - 25.0:
        return [23, 15]
    
    # Step 2: Compute sigma(n) using SPF-based multiplicative recurrence
    # sigma is multiplicative. For n = p^k * r with gcd(p,r)=1:
    # sigma(n) = sigma(r) * (p^(k+1)-1)/(p-1)
    # 
    # We compute sigma incrementally:
    # sigma[1] = 1
    # For n >= 2: let p = spf[n], m = n // p
    #   if m % p == 0: p^2 | n, let p^k || n, p^(k-1) || m
    #     sigma(n) = sigma(m) - p^(k-1) + p^k  ... NO, this is wrong
    #     Correct: sigma(n) = sigma(m) + p^k * sigma(n / p^k)
    #     But we need sigma(n/p^k). Let's track differently.
    #
    # Better approach: track for each n:
    #   p_pow[n] = p^k where p = spf[n] and p^k || n
    #   sigma_r[n] = sigma(n / p^k) = sigma of the part coprime to p
    # Then sigma[n] = sigma_r[n] * (p^(k+1)-1)//(p-1)
    #
    # Recurrence:
    #   p = spf[n], m = n // p
    #   if m % p == 0:
    #       p_pow[n] = p_pow[m] * p
    #       sigma_r[n] = sigma_r[m]
    #   else:
    #       p_pow[n] = p
    #       sigma_r[n] = sigma[m]
    #   sigma[n] = sigma_r[n] * (p_pow[n] * (p_pow[n]*p - 1)) // (p - 1)
    #             = sigma_r[n] * (p^(k+1) - 1) // (p - 1)
    
    sigma = np.ones(N + 1, dtype=np.int64)
    p_pow = np.ones(N + 1, dtype=np.int64)  # p_pow[n] = p^k where p=spf[n], p^k||n
    sigma_r = np.ones(N + 1, dtype=np.int64)  # sigma_r[n] = sigma(n/p^k)
    
    for n in range(2, N + 1):
        if time.time() > deadline - 20.0:
            break
        p = int(spf[n])
        m = n // p
        if m % p == 0:
            p_pow[n] = p_pow[m] * p
            sigma_r[n] = sigma_r[m]
        else:
            p_pow[n] = p
            sigma_r[n] = sigma[m]
        # sigma(n) = sigma_r[n] * (p^(k+1) - 1) / (p - 1)
        # p^(k+1) - 1 = p_pow[n] * p - 1
        # (p^(k+1) - 1) / (p - 1) = 1 + p + ... + p^k
        # Let pk = p_pow[n]
        # (pk * p - 1) // (p - 1)
        pk = int(p_pow[n])
        sigma[n] = sigma_r[n] * (pk * p - 1) // (p - 1)
    
    if time.time() > deadline - 5.0:
        return [23, 15]
    
    # Step 3: Group numbers by sigma value and find best pair
    sig = sigma[1:N+1]
    nums = np.arange(1, N+1, dtype=np.int64)
    
    # Sort by sigma value
    order = np.argsort(sig, kind='stable')
    sorted_sig = sig[order]
    sorted_nums = nums[order]
    
    # Find groups with same sigma value
    changes = np.concatenate(([True], sorted_sig[1:] != sorted_sig[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [len(sorted_sig)]))
    
    best_err = float('inf')
    best_pair = (23, 15)
    
    for gi in range(len(group_starts)):
        if time.time() > deadline - 2.0:
            break
        
        gs = group_starts[gi]
        ge = group_ends[gi]
        group_size = ge - gs
        
        if group_size < 2:
            continue
        
        group_nums = sorted_nums[gs:ge]
        
        # For each b in group_nums, find a in group_nums closest to 3b/2
        targets = 1.5 * group_nums.astype(np.float64)
        insert_pos = np.searchsorted(group_nums, targets, side='left')
        
        ip = np.clip(insert_pos, 0, len(group_nums) - 1)
        ip_minus = np.clip(insert_pos - 1, 0, len(group_nums) - 1)
        
        a_cand1 = group_nums[ip]
        a_cand2 = group_nums[ip_minus]
        
        b_vals = group_nums
        num1 = np.abs(2 * a_cand1 - 3 * b_vals)
        den1 = 2.0 * b_vals.astype(np.float64)
        err1 = num1 / den1
        
        num2 = np.abs(2 * a_cand2 - 3 * b_vals)
        den2 = 2.0 * b_vals.astype(np.float64)
        err2 = num2 / den2
        
        err = np.minimum(err1, err2)
        
        mask1 = (a_cand1 != b_vals)
        mask2 = (a_cand2 != b_vals)
        
        final_a = np.where(mask1, a_cand1, a_cand2)
        final_err = np.where(mask1, err1, err2)
        
        both_same = (~mask1) & (~mask2)
        final_err[both_same] = float('inf')
        
        min_idx = np.argmin(final_err)
        min_err = final_err[min_idx]
        
        if min_err < best_err:
            min_a = int(final_a[min_idx])
            min_b = int(b_vals[min_idx])
            best_err = min_err
            best_pair = (min_a, min_b)
    
    return [int(best_pair[0]), int(best_pair[1])]
