import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Generate primes up to 10^6 (sqrt(10^12))
    limit = 10**6
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # 2. Define segment size and search strategy
    # We will scan segments of size 5,000,000.
    # We'll check a few segments in the lower range first, as phi values are smaller 
    # and might have different collision properties, but usually larger n is fine.
    # Actually, let's just scan consecutive segments starting from 1.
    # If we find a very long run (e.g. > 50), we might want to verify or just keep searching.
    
    seg_size = 5 * 10**6
    num_segments_to_try = 20 # Rough estimate of what fits in time
    
    for seg_idx in range(num_segments_to_try):
        if time.time() > deadline:
            break
            
        L = seg_idx * seg_size + 1
        R = L + seg_size - 1
        
        if R > 10**12:
            R = 10**12
            L = max(1, R - seg_size + 1)
            
        # Compute phi for [L, R]
        size = R - L + 1
        if size <= 0:
            continue
            
        # Initialize phi values
        # Using float64 to avoid integer overflow issues during intermediate steps? 
        # phi(n) < 10^12, which fits in int64. But operations like x // p are integer.
        # Let's use int64.
        phi_vals = np.arange(L, R + 1, dtype=np.int64)
        
        # Apply sieve updates
        for p in primes:
            if p > R:
                break
            # Find first multiple of p in [L, R]
            # first_mult = ceil(L / p) * p
            first_mult = ((L + p - 1) // p) * p
            
            if first_mult > R:
                continue
                
            # Indices in phi_vals corresponding to multiples of p
            # We can generate indices: first_mult - L, first_mult + p - L, ...
            # Using np.arange is efficient
            idx = np.arange(first_mult - L, size, p)
            
            # Update phi: phi[n] = phi[n] * (1 - 1/p) = phi[n] - phi[n]//p
            # Vectorized operation
            phi_vals[idx] -= phi_vals[idx] // p
            
        # Find longest run of distinct values in phi_vals
        # Sliding window
        seen = {}
        left = 0
        max_k_seg = 0
        max_start_seg = 0
        
        # Optimization: We only care about runs up to 200.
        # But we don't know if there are longer ones. The task says k <= 200.
        # So we can cap the window size at 200?
        # If a window of 200 is distinct, that's the max possible answer.
        # We can stop searching this segment if we find 200.
        
        for right in range(size):
            val = phi_vals[right]
            
            # If val is already in seen, move left until it's not
            while val in seen and left <= right:
                # Remove left element
                l_val = phi_vals[left]
                seen[l_val] -= 1
                if seen[l_val] == 0:
                    del seen[l_val]
                left += 1
                
            # Add current val
            if val in seen:
                seen[val] += 1
            else:
                seen[val] = 1
                
            # Update max
            cur_k = right - left + 1
            if cur_k > max_k_seg:
                max_k_seg = cur_k
                max_start_seg = left
                if max_k_seg == 200:
                    break # Can't do better in this segment
            
        # Update global best
        if max_k_seg > best_k:
            best_k = max_k_seg
            best_s = L + max_start_seg
            
            # If we hit the cap, we can't do better. 
            # But maybe we should verify? The task says maximize k. 200 is the max.
            # If best_k == 200, we can return immediately? 
            # The problem says 1 <= k <= 200. So 200 is the global max.
            if best_k == 200:
                return {'start': best_s, 'length': best_k}

    return {'start': best_s, 'length': best_k}
