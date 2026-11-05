import numpy as np
import time
import random

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    W = 200
    
    while time.time() - start_time < time_limit:
        # Pick a random start
        # Ensure s + W - 1 <= 10^12
        max_s = 10**12 - W + 1
        if max_s < 1:
            break
        s = random.randint(1, max_s)
        
        # Compute phi for [s, s+W-1]
        # phi_arr
        phi_arr = np.arange(s, s + W, dtype=np.int64)
        residual_arr = np.arange(s, s + W, dtype=np.int64)
        
        # Sieve pass
        for p in primes:
            if p > s + W - 1: # Optimization: if p > max_n, no multiples? No, p can be > max_n? No, p <= 10^6.
                # Actually if p > s+W-1, it can't divide any number in [s, s+W-1] unless the number is p itself.
                # But p <= 10^6. s can be large.
                # We need to find multiples.
                pass
            
            # Find first multiple of p in [s, s+W-1]
            start_idx = ((s + p - 1) // p) * p
            if start_idx > s + W - 1:
                continue
            
            # Indices in the array
            # start_idx - s, start_idx - s + p, ...
            idxs = np.arange(start_idx - s, W, p)
            
            phi_arr[idxs] = phi_arr[idxs] // p * (p - 1)
            residual_arr[idxs] = residual_arr[idxs] // p
            
        # Cleanup pass
        for i in range(W):
            r = residual_arr[i]
            if r == 1:
                continue
            
            # Remove small factors from r
            # We only need to check primes up to sqrt(r)
            # But r can be up to 10^12. sqrt(r) up to 10^6.
            # We have primes up to 10^6.
            for p in primes:
                if p * p > r:
                    break
                if r % p == 0:
                    while r % p == 0:
                        r //= p
            
            if r > 1 and r > 10**6:
                # r is a large prime factor
                # phi_arr[i] is currently n * prod(1-1/p_small)
                # We need to multiply by (1-1/r)
                phi_arr[i] = phi_arr[i] // r * (r - 1)
        
        # Check for longest run of distinct values
        # O(W^2) check
        max_run = 0
        max_start_offset = 0
        for start_off in range(W):
            seen = set()
            for end_off in range(start_off, W):
                val = int(phi_arr[end_off])
                if val in seen:
                    break
                seen.add(val)
                run_len = end_off - start_off + 1
                if run_len > max_run:
                    max_run = run_len
                    max_start_offset = start_off
        
        if max_run > best_k:
            best_k = max_run
            best_s = s + max_start_offset
            if best_k >= 200:
                return {'start': best_s, 'length': best_k}
        
        # If time is running out, break
        if time.time() - start_time > time_limit - 5:
            break

    return {'start': best_s, 'length': best_k}
