import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s kill
    
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 1
    best_s = 1
    
    # Define a function to process a segment
    def process_segment(L, R):
        nonlocal best_k, best_s
        if R > 10**12:
            R = 10**12
        if L > R:
            return
        n = R - L + 1
        phi = np.arange(L, R+1, dtype=np.int64)
        temp = np.arange(L, R+1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            start = L if L % p == 0 else L + p - L % p
            if start > R:
                continue
            indices = np.arange(start, R+1, p) - L
            phi[indices] = phi[indices] // p * (p - 1)
            
            t = temp[indices].copy()
            t //= p
            # Remove remaining factors of p
            # This loop might be slow if done in Python, but p is small
            # For p=2, t can be divided by 2 many times
            while True:
                mask = t % p == 0
                if not np.any(mask):
                    break
                t[mask] //= p
            temp[indices] = t
        
        # Handle remaining prime factors
        mask = temp > 1
        if np.any(mask):
            q = temp[mask]
            phi[mask] = phi[mask] // q * (q - 1)
        
        # Find max run of distinct phi values
        counts = {}
        left = 0
        max_len = 0
        max_start = 0
        for right in range(n):
            val = phi[right]
            counts[val] = counts.get(val, 0) + 1
            while counts[val] > 1:
                left_val = phi[left]
                counts[left_val] -= 1
                left += 1
            current_len = right - left + 1
            if current_len > max_len:
                max_len = current_len
                max_start = left
        
        if max_len > best_k:
            best_k = max_len
            best_s = L + max_start
            
        return best_k, best_s

    # Search strategy
    # 1. Small ranges
    step = 10**6
    for L in range(1, 10**7 + 1, step):
        if time.time() > deadline:
            break
        process_segment(L, L + step - 1)
        
    # 2. Large ranges
    large_starts = [10**12 - 10**6, 10**11, 5 * 10**11, 10**10]
    for L in large_starts:
        if time.time() > deadline:
            break
        process_segment(L, L + 10**6 - 1)

    return {'start': best_s, 'length': best_k}
