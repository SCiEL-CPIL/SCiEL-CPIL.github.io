import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    best_k = 1
    best_s = 1
    
    W = 2 * 10**6
    
    random.seed(42)
    
    max_L = 10**12 - W
    
    # Pre-allocate arrays if possible? 
    # L changes, so we can't pre-allocate the values, but we can pre-allocate the arrays and fill them?
    # np.arange is fast.
    
    for _ in range(200): # Increased iterations
        if time.time() > deadline - 5.0:
            break
            
        L = random.randint(1, max_L)
        
        # Compute phi
        phi = np.arange(L, L + W, dtype=np.int64)
        tmp = phi.copy()
        
        for p in primes:
            if p > L + W - 1:
                break
            # Check if any element in tmp is divisible by p
            # Using vectorized modulo
            mask = (tmp % p == 0)
            if np.any(mask):
                phi[mask] = phi[mask] * (p - 1) // p
                tmp[mask] = tmp[mask] // p
                
        # Handle remaining factors
        mask = (tmp > 1)
        if np.any(mask):
            phi[mask] = phi[mask] * (tmp[mask] - 1) // tmp[mask]
            
        # Find longest distinct run
        # Convert to list for efficient Python set operations
        phi_list = phi.tolist()
        
        seen = set()
        left = 0
        current_max_k = 0
        
        for right in range(W):
            val = phi_list[right]
            while val in seen:
                seen.discard(phi_list[left])
                left += 1
                if left > right:
                    break
            seen.add(val)
            current_window_len = right - left + 1
            if current_window_len > current_max_k:
                current_max_k = current_window_len
                if current_max_k > best_k:
                    best_k = current_max_k
                    best_s = L + left
                    if best_k == 200:
                        return {'start': best_s, 'length': best_k}
                        
        if best_k == 200:
            break

    return {'start': best_s, 'length': best_k}
