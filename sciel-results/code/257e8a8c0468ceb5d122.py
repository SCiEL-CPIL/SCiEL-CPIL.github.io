import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Sieve of Eratosthenes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Segmented Sieve for Phi
    W = 10**6 
    max_k = 200
    best_k = 1
    best_s = 1
    
    prev_phi_tail = []
    prev_start = 0
    
    current_start = 1
    
    while current_start <= 10**12:
        if time.time() > deadline:
            break
            
        L = current_start
        R = min(L + W - 1, 10**12)
        seg_len = R - L + 1
        
        phi_vals = [0] * seg_len
        rem_vals = [0] * seg_len
        
        for i in range(seg_len):
            val = L + i
            phi_vals[i] = val
            rem_vals[i] = val
            
        for p in primes:
            if p * p > R:
                break
            first = ((L + p - 1) // p) * p
            for m in range(first, R + 1, p):
                idx = m - L
                phi_vals[idx] -= phi_vals[idx] // p
                while rem_vals[idx] % p == 0:
                    rem_vals[idx] //= p
        
        for i in range(seg_len):
            if rem_vals[i] > 1:
                phi_vals[i] -= phi_vals[i] // rem_vals[i]
        
        combined = prev_phi_tail + phi_vals
        
        seen = {}
        left = 0
        max_len_in_seg = 0
        max_start_in_seg = 0
        
        for right in range(len(combined)):
            val = combined[right]
            if val in seen:
                seen[val] += 1
            else:
                seen[val] = 1
            
            while seen[val] > 1:
                left_val = combined[left]
                seen[left_val] -= 1
                left += 1
                
            curr_len = right - left + 1
            if curr_len > max_len_in_seg:
                max_len_in_seg = curr_len
                max_start_in_seg = left
        
        if max_len_in_seg > best_k:
            if max_start_in_seg < len(prev_phi_tail):
                global_start = prev_start + max_start_in_seg
            else:
                global_start = L + (max_start_in_seg - len(prev_phi_tail))
            
            reported_k = min(max_len_in_seg, 200)
            if reported_k > best_k:
                best_k = reported_k
                best_s = global_start
                
        if seg_len >= max_k:
            prev_phi_tail = phi_vals[-max_k:]
            prev_start = R - max_k + 1
        else:
            prev_phi_tail = phi_vals[:]
            prev_start = L
            
        current_start = R + 1
        
    return {'start': best_s, 'length': best_k}
