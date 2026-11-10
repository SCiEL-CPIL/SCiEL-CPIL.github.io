import numpy as np
import time
import sys
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    N = 10**7
    
    # Linear sieve for sigma(n), n=1..N
    sigma = [0] * (N + 1)
    sigma[1] = 1
    power = [0] * (N + 1)
    sigma_power = [0] * (N + 1)
    is_prime = [True] * (N + 1)
    primes = []
    
    sig = sigma
    pow_ = power
    spow = sigma_power
    isp = is_prime
    prm = primes
    N_local = N
    prm_append = prm.append
    
    for i in range(2, N_local + 1):
        if isp[i]:
            prm_append(i)
            sig[i] = 1 + i
            pow_[i] = i
            spow[i] = 1 + i
        
        for p in prm:
            ip = i * p
            if ip > N_local:
                break
            isp[ip] = False
            if i % p == 0:
                pow_[ip] = pow_[i] * p
                spow[ip] = spow[i] + pow_[ip]
                sig[ip] = sig[i] // spow[i] * spow[ip]
                break
            else:
                pow_[ip] = p
                spow[ip] = 1 + p
                sig[ip] = sig[i] * (1 + p)
                
    if time.time() > deadline - 25:
        return [11, 6]

    sigma_np = np.array(sig, dtype=np.int64)
    sorted_indices = np.argsort(sigma_np, kind='stable')
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Vectorized group boundary detection
    sigma_sorted = sigma_np[sorted_indices]
    change_points = np.where(sigma_sorted[1:] != sigma_sorted[:-1])[0] + 1
    
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    prev = 0
    n_sorted = len(sorted_indices)
    
    for cp in change_points:
        if time.time() > deadline - 3:
            break
        
        group_size = cp - prev
        if group_size >= 2:
            nums = sorted_indices[prev:cp]
            
            if group_size < 1000:
                nums_list = nums.tolist()
                nums_list.sort()
            else:
                nums_list = np.sort(nums).tolist()
                
            min_val = nums_list[0]
            max_val = nums_list[-1]
            
            ratio_min = min_val / max_val
            ratio_max = max_val / min_val
            
            if ratio_max < 1.5:
                potential_err = 1.5 - ratio_max
            elif ratio_min > 1.5:
                potential_err = ratio_min - 1.5
            else:
                potential_err = 0.0
                
            if potential_err < best_err:
                for b in nums_list:
                    target_a = 1.5 * b
                    pos = bisect.bisect_left(nums_list, target_a)
                    
                    if pos < group_size:
                        a = nums_list[pos]
                        if a != b:
                            err = abs(a/b - 1.5)
                            if err < best_err and err != 0:
                                best_err = err
                                best_a, best_b = a, b
                    if pos - 1 >= 0:
                        a = nums_list[pos - 1]
                        if a != b:
                            err = abs(a/b - 1.5)
                            if err < best_err and err != 0:
                                best_err = err
                                best_a, best_b = a, b
        prev = cp
    
    # Handle last group
    if prev < n_sorted:
        group_size = n_sorted - prev
        if group_size >= 2:
            nums = sorted_indices[prev:]
            if group_size < 1000:
                nums_list = nums.tolist()
                nums_list.sort()
            else:
                nums_list = np.sort(nums).tolist()
            min_val = nums_list[0]
            max_val = nums_list[-1]
            ratio_min = min_val / max_val
            ratio_max = max_val / min_val
            if ratio_max < 1.5:
                potential_err = 1.5 - ratio_max
            elif ratio_min > 1.5:
                potential_err = ratio_min - 1.5
            else:
                potential_err = 0.0
            if potential_err < best_err:
                for b in nums_list:
                    target_a = 1.5 * b
                    pos = bisect.bisect_left(nums_list, target_a)
                    if pos < group_size:
                        a = nums_list[pos]
                        if a != b:
                            err = abs(a/b - 1.5)
                            if err < best_err and err != 0:
                                best_err = err
                                best_a, best_b = a, b
                    if pos - 1 >= 0:
                        a = nums_list[pos - 1]
                        if a != b:
                            err = abs(a/b - 1.5)
                            if err < best_err and err != 0:
                                best_err = err
                                best_a, best_b = a, b

    return [int(best_a), int(best_b)]
