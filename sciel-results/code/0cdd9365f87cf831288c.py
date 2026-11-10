import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
        
    # Sieve primes up to sqrt(10^12) = 10^6
    limit = 10**6 + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    best_k = 1
    best_s = 1
    
    # Use larger block size for better throughput
    W = 10 * 10**6
    
    lo = 1
    while lo <= 10**12:
        if time_left() < 15.0:
            break
            
        hi = min(lo + W - 1, 10**12)
        n = hi - lo + 1
        
        # Initialize phi and rem
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        rem = np.arange(lo, hi + 1, dtype=np.int64)
        
        # Process primes
        max_p = int(math.isqrt(hi))
        idx_end = np.searchsorted(primes, max_p + 1)
        relevant_primes = primes[:idx_end]
        
        for p in relevant_primes:
            p = int(p)
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
                
            indices = np.arange(start, hi + 1, p) - lo
            if len(indices) == 0:
                continue
                
            # Apply phi correction: phi[n] *= (p-1)/p
            phi[indices] = phi[indices] // p * (p - 1)
            
            # Divide p out of rem completely using vectorized approach
            r = rem[indices].copy()
            while True:
                mask = r % p == 0
                if not np.any(mask):
                    break
                r[mask] //= p
            rem[indices] = r
            
        # Handle remaining prime factors
        mask = rem > 1
        if np.any(mask):
            phi[mask] = phi[mask] // rem[mask] * (rem[mask] - 1)
            
        # Find longest run of distinct values using last_seen dict
        last_seen = {}
        start_idx = 0
        for i in range(len(phi)):
            v = phi[i]
            if v in last_seen:
                if last_seen[v] >= start_idx:
                    start_idx = last_seen[v] + 1
            last_seen[v] = i
            k = i - start_idx + 1
            if k > best_k:
                best_k = k
                best_s = lo + start_idx
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}
            
        lo = hi + 1
        
    return {'start': best_s, 'length': best_k}
