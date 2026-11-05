"""Sieve-based phi computation with sliding window for long distinct-totient runs."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    MAX_N = 10**12
    MAX_K = 200
    WINDOW = 2_000_000  # size of segmented sieve window
    
    # Sieve primes up to sqrt(MAX_N)
    limit = int(MAX_N**0.5) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # Search multiple windows across the range
    # Focus on regions where numbers are "rough" (have large prime factors)
    # to spread out phi values
    num_windows = 30
    for wi in range(num_windows):
        # Spread windows across the range, biased toward larger values
        # where phi values tend to be larger and more distinct
        if wi == 0:
            start = 1
        else:
            # Exponential spacing toward larger values
            frac = (wi / num_windows) ** 2
            start = int(1 + frac * (MAX_N - WINDOW))
            start = max(1, start)
        
        end = min(start + WINDOW - 1, MAX_N)
        if end - start + 1 < MAX_K:
            continue
        
        n = end - start + 1
        
        # Segmented phi sieve
        phi_vals = np.arange(start, end + 1, dtype=np.int64).copy()
        remaining = phi_vals.copy()
        
        for p in primes:
            if p * p > end:
                break
            # Find multiples of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first < start:
                first = start
            indices = np.arange(first, end + 1, p)
            if len(indices) == 0:
                continue
            local_idx = indices - start
            # For each multiple, check if p divides it
            for idx in local_idx:
                if remaining[idx] % p == 0:
                    phi_vals[idx] -= phi_vals[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors (primes > sqrt(end))
        for i in range(n):
            if remaining[i] > 1:
                phi_vals[i] -= phi_vals[i] // remaining[i]
        
        # Find longest run of distinct phi values using sliding window
        # Use a set-based approach with two pointers
        seen = set()
        left = 0
        current_max = 0
        for right in range(n):
            v = phi_vals[right]
            if v in seen:
                # Shrink from left
                while left <= right:
                    seen.discard(phi_vals[left])
                    left += 1
                    if left > right:
                        break
            seen.add(v)
            cur_len = right - left + 1
            if cur_len > current_max:
                current_max = cur_len
                if current_max > best_k:
                    best_k = current_max
                    best_s = start + left
        
        if best_k >= MAX_K:
            break
    
    # Verify and clamp
    if best_s + best_k - 1 > MAX_N:
        best_k = int(MAX_N - best_s + 1)
    if best_k > MAX_K:
        best_k = MAX_K
    if best_k < 1:
        best_k = 1
        best_s = 1
    
    return {'start': int(best_s), 'length': int(best_k)}
# EVOLVE-BLOCK-END
