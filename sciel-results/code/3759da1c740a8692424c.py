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
    
    # Use smaller block size to reduce memory and improve time management
    W = 3 * 10**6
    
    # Strategy: search in multiple ranges, prioritizing larger values where
    # longer distinct-totient runs are more likely
    # First try a range near 10^12, then work backwards, then fill gaps
    
    # Define search ranges to try (prioritize larger values)
    search_ranges = []
    max_val = 10**12
    
    # Try ranges starting from high values
    num_ranges = 50
    range_size = max_val // num_ranges
    
    for i in range(num_ranges):
        hi = max_val - i * range_size
        lo = max(1, hi - range_size + 1)
        search_ranges.append((lo, hi))
    
    # Also add some smaller ranges for completeness
    search_ranges.append((1, W))
    
    for lo, hi in search_ranges:
        if time_left() < 15.0:
            break
            
        # Process this range in chunks
        current_lo = lo
        while current_lo <= hi:
            if time_left() < 10.0:
                break
                
            current_hi = min(current_lo + W - 1, hi)
            n = current_hi - current_lo + 1
            
            # Initialize phi and rem
            phi = np.arange(current_lo, current_hi + 1, dtype=np.int64)
            rem = np.arange(current_lo, current_hi + 1, dtype=np.int64)
            
            # Process primes
            max_p = int(math.isqrt(current_hi))
            idx_end = np.searchsorted(primes, max_p + 1)
            relevant_primes = primes[:idx_end]
            
            for p in relevant_primes:
                p = int(p)
                start = ((current_lo + p - 1) // p) * p
                if start > current_hi:
                    continue
                    
                indices = np.arange(start, current_hi + 1, p) - current_lo
                
                phi[indices] -= phi[indices] // p
                
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
                phi[mask] -= phi[mask] // rem[mask]
                
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
                    best_s = current_lo + start_idx
                    if best_k == 200:
                        return {'start': best_s, 'length': best_k}
                    
            current_lo = current_hi + 1
            
    return {'start': best_s, 'length': best_k}
