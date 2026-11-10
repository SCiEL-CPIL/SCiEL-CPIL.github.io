import math
import numpy as np
from collections import defaultdict

def construct():
    # New Approach: Generate smooth numbers and compute sigma directly.
    # This targets a specific subset of k's (smooth numbers) which allows 
    # efficient computation and potentially finding M with more collisions.
    
    # Primes to use for smoothness
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Limit for k. 
    # We generate all 47-smooth numbers up to this limit.
    LIMIT_K = 10_000_000
    
    groups = defaultdict(list)
    
    # Helper to compute sigma for a smooth number using the given primes
    def sigma_smooth(n):
        sigma_val = 1
        temp = n
        for p in primes:
            if temp % p == 0:
                exp = 0
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    p_power *= p
                # sum of geometric series: (p^(exp+1) - 1) / (p - 1)
                sigma_factor = (p_power * p - 1) // (p - 1)
                sigma_val *= sigma_factor
        if temp > 1:
            # n had a prime factor > last prime in list, so it's not fully smooth
            return None
        return sigma_val

    # Recursive generation of smooth numbers
    def dfs(idx, val):
        if val > LIMIT_K:
            return
        
        # This val is a smooth number
        s = sigma_smooth(val)
        if s is not None:
            M = val * s
            if M <= 10**18:
                groups[M].append(val)
        
        # Extend with primes from idx onwards
        for j in range(idx, len(primes)):
            p = primes[j]
            next_val = val * p
            if next_val > LIMIT_K:
                break
            dfs(j, next_val)

    dfs(0, 1)
    
    best_M = None
    best_ks = []
    
    for M, ks in groups.items():
        sqrt_M = math.isqrt(M)
        # Filter k <= sqrt_M
        valid_ks = [k for k in ks if k <= sqrt_M]
        if len(valid_ks) > len(best_ks):
            best_ks = valid_ks
            best_M = M
            
    if best_M is None:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
