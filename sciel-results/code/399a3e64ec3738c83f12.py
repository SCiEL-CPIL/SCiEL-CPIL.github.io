import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    MIN_N = 10000
    MAX_N = 1000000

    def get_smooth_numbers(primes, low, high):
        """Generate all numbers in [low, high] whose prime factors are in primes."""
        results = set()
        
        def dfs(idx, current_val):
            if current_val > high:
                return
            if current_val >= low:
                results.add(current_val)
            
            if idx == len(primes):
                return
                
            p = primes[idx]
            next_val = current_val
            while next_val <= high:
                dfs(idx + 1, next_val)
                if next_val > high // p:
                    break
                next_val *= p
        
        dfs(0, 1)
        return results

    # Check for smooth pairs with increasing P_max
    # We stop as soon as we find a valid pair.
    prime_sets = [
        [2],
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    for primes in prime_sets:
        if time.time() > deadline - 10:
            break
            
        smooths = get_smooth_numbers(primes, MIN_N, MAX_N + 1)
        
        if not smooths:
            continue
            
        smooth_list = sorted(smooths)
        found_n = None
        
        for n in smooth_list:
            # n is a smooth number in [MIN_N, MAX_N+1]
            # We need n in [MIN_N, MAX_N] for it to be a valid candidate n
            if n < MIN_N or n > MAX_N:
                continue
                
            # Check if n-1 is smooth (so max(LPF(n), LPF(n-1)) is small? No, we need n and n+1)
            # Wait, the pair is (n, n+1).
            # If n is smooth, we check if n+1 is smooth.
            # If n+1 is smooth, then n is a valid candidate.
            if (n + 1) in smooths:
                found_n = n
                break
            # Also check if n-1 is smooth? 
            # If n-1 is smooth and n is smooth, then n-1 is a valid candidate.
            # Since we are iterating n in increasing order, if n-1 was a valid candidate, 
            # it would have been found when we processed n-1 (checking n-1 + 1 = n).
            # So we only need to check n+1.
            
        if found_n is not None:
            return found_n
            
    # If no pair found with P_max <= 13, we fall back to the full scan
    MAX_N_LIMIT = 1000000
    LIMIT = MAX_N_LIMIT + 1
    
    spf = np.arange(LIMIT + 1, dtype=np.int32)
    spf[0] = 0
    spf[1] = 1
    
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:
            start = i * i
            step = i
            spf[start:LIMIT+1:step] = np.minimum(spf[start:LIMIT+1:step], i)
            
    lpf = np.ones(LIMIT + 1, dtype=np.int32)
    for n in range(2, LIMIT + 1):
        m = n
        while m > 1:
            p = spf[m]
            lpf[n] = p
            while m % p == 0:
                m //= p
                
    n_range = np.arange(MIN_N, MAX_N_LIMIT + 1)
    lpf_n = lpf[n_range]
    lpf_n1 = lpf[n_range + 1]
    current_lpf = np.maximum(lpf_n, lpf_n1)
    
    best_idx = np.argmin(current_lpf)
    best_n = int(n_range[best_idx])
    
    return best_n
# EVOLVE-BLOCK-END
