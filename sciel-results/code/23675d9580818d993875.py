import numpy as np
import time
import random
import math

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
    primes_list = primes.tolist()
    
    # Miller-Rabin for primality testing of large numbers
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        
        # Deterministic bases for n < 3.3 * 10^24
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    best_k = 1
    best_s = 1
    W = 200
    
    # Helper to compute phi for a window using vectorized sieve
    def compute_phi_window(s, W):
        phi_arr = np.arange(s, s + W, dtype=np.int64)
        residual_arr = np.arange(s, s + W, dtype=np.int64)
        
        # Sieve pass
        for p in primes:
            p_int = int(p)
            # Find first multiple of p in [s, s+W-1]
            start_idx = ((s + p_int - 1) // p_int) * p_int
            if start_idx > s + W - 1:
                continue
            
            # Indices in the array
            idxs = np.arange(start_idx - s, W, p_int)
            
            phi_arr[idxs] = phi_arr[idxs] // p_int * (p_int - 1)
            residual_arr[idxs] = residual_arr[idxs] // p_int
        
        # Cleanup pass
        for i in range(W):
            r = int(residual_arr[i])
            if r == 1:
                continue
            
            # Remove small factors from r
            for p in primes_list:
                if p * p > r:
                    break
                if r % p == 0:
                    while r % p == 0:
                        r //= p
            
            if r > 1:
                # r is either 1 or a prime (since n <= 10^12)
                # Use Miller-Rabin to confirm it's prime
                if r > 10**6:
                    if is_prime_mr(r):
                        phi_arr[i] = int(phi_arr[i]) // r * (r - 1)
                    else:
                        # This shouldn't happen for n <= 10^12, but handle it
                        # r is composite, need to factor it further
                        # For safety, just treat it as prime (shouldn't occur)
                        phi_arr[i] = int(phi_arr[i]) // r * (r - 1)
                else:
                    phi_arr[i] = int(phi_arr[i]) // r * (r - 1)
        
        return phi_arr
    
    # Helper to find longest run of distinct values
    def find_longest_run(phi_arr):
        max_run = 0
        max_start_offset = 0
        W_local = len(phi_arr)
        for start_off in range(W_local):
            seen = set()
            for end_off in range(start_off, W_local):
                val = int(phi_arr[end_off])
                if val in seen:
                    break
                seen.add(val)
                run_len = end_off - start_off + 1
                if run_len > max_run:
                    max_run = run_len
                    max_start_offset = start_off
        return max_run, max_start_offset
    
    # Search loop
    while time.time() - start_time < time_limit - 5.0:
        # Pick a random start, biased toward larger numbers
        # Larger numbers tend to have more distinct totients
        if random.random() < 0.7:
            # 70% chance: large numbers
            s = random.randint(10**10, 10**12 - W + 1)
        else:
            # 30% chance: uniform
            s = random.randint(1, 10**12 - W + 1)
        
        # Compute phi for [s, s+W-1]
        phi_arr = compute_phi_window(s, W)
        
        # Find longest run of distinct values
        max_run, max_start_offset = find_longest_run(phi_arr)
        
        if max_run > best_k:
            best_k = max_run
            best_s = s + max_start_offset
            if best_k >= 200:
                return {'start': best_s, 'length': best_k}
    
    return {'start': best_s, 'length': best_k}
