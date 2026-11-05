import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # 2. Segmented Sieve for Phi
    SEG_SIZE = 10**6
    MAX_N = 10**9  # Search up to 10^9
    
    best_k = 0
    best_s = 1
    
    # State carried over between segments
    current_run_start = 1
    seen_vals = set()
    
    L = 1
    while L <= MAX_N:
        if time.time() - start_time > time_limit:
            break
            
        R = min(L + SEG_SIZE - 1, MAX_N)
        n = R - L + 1
        
        # Initialize arrays
        phi_arr = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply phi sieve
        for p in primes:
            if p * p > R:
                break
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            idxs = np.arange(start - L, n, p)
            phi_arr[idxs] -= phi_arr[idxs] // p
            
        # Check for collisions
        for i in range(n):
            val = int(phi_arr[i])
            curr_num = L + i
            
            if val in seen_vals:
                # Collision! Run ends at curr_num - 1
                run_len = (curr_num - 1) - current_run_start + 1
                if run_len > best_k:
                    best_k = run_len
                    best_s = current_run_start
                    if best_k >= 200:
                        return {'start': best_s, 'length': best_k}
                
                # Start new run
                current_run_start = curr_num
                seen_vals = {val}
            else:
                seen_vals.add(val)
                if curr_num - current_run_start + 1 > best_k:
                    best_k = curr_num - current_run_start + 1
                    best_s = current_run_start
                    if best_k >= 200:
                        return {'start': best_s, 'length': best_k}
        
        L = R + 1
        
    # Handle the last run if it extends to the end of the search range
    if L > 1:
        run_len = (L - 1) - current_run_start + 1
        if run_len > best_k:
            best_k = run_len
            best_s = current_run_start

    return {'start': best_s, 'length': best_k}
