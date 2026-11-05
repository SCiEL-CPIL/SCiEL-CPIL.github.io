import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
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
    
    # We will scan windows of size W
    W = 2 * 10**6  # 2 million numbers per scan
    
    # Candidate starting points
    # We want to scan large regions.
    # Random sampling across the range [1, 10^12]
    random.seed(42)
    
    # We can afford to scan maybe 5-10 windows of size W in 115s?
    # Let's estimate time for one window.
    # Operations ~ W * sum(1/p) ~ 2.6 * W.
    # For W=2e6, ops ~ 5.2e6.
    # Python can do ~10-50 million simple ops per second.
    # So one window should take < 1 second.
    # We can try many windows.
    
    # Strategy: Pick random starts, ensure they don't overlap too much, 
    # and cover the range.
    
    # To be safe and diverse, let's pick starts spaced out or random.
    # Since collisions are rare, random is good.
    
    # Let's try 10 random windows.
    for _ in range(20):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random start L
        # Max L is 10^12 - W
        max_L = 10**12 - W
        if max_L < 1:
            break
        L = random.randint(1, max_L)
        
        # Compute phi for L to L+W-1
        phi = [0] * W
        tmp = [0] * W
        
        for i in range(W):
            phi[i] = L + i
            tmp[i] = L + i
            
        # Sieve
        for p in primes:
            if p * p > L + W - 1:
                # Optimization: if p*p > max_val, we only need to check if tmp[idx] % p == 0 
                # but since tmp[idx] < p^2, if it's divisible by p, it must be p itself or 1.
                # Actually, if p > sqrt(max_val), then p can appear at most once.
                # The standard segmented sieve logic holds.
                pass
            
            # Find first multiple of p in [L, L+W-1]
            # first_multiple = ceil(L / p) * p
            first_multiple = ((L + p - 1) // p) * p
            
            # Iterate
            for m in range(first_multiple, L + W, p):
                idx = m - L
                if tmp[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while tmp[idx] % p == 0:
                        tmp[idx] //= p
        
        # Handle remaining factors
        for i in range(W):
            if tmp[i] > 1:
                phi[i] -= phi[i] // tmp[i]
        
        # Sliding window for distinct phi values
        seen = set()
        left = 0
        current_max_k = 0
        
        for right in range(W):
            val = phi[right]
            while val in seen:
                # Remove left element
                seen.discard(phi[left])
                left += 1
                if left > right:
                    break
            seen.add(val)
            current_window_len = right - left + 1
            if current_window_len > current_max_k:
                current_max_k = current_window_len
                # We could record best_s here, but we need global best
                if current_max_k > best_k:
                    best_k = current_max_k
                    best_s = L + left
                    # If we hit 200, we can't do better
                    if best_k == 200:
                        return {'start': best_s, 'length': best_k}
                        
        if best_k == 200:
            break

    return {'start': best_s, 'length': best_k}
