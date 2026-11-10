"""Optimized program for few distinct small-prime parts in short integer interval."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max value we need to precompute: n+24 where n <= 1000000
    max_val = 1000024
    
    # Precompute small-prime factor for all numbers using vectorized sieve
    # sf[x] = product of prime factors p < 24 (with multiplicity)
    sf = np.ones(max_val + 1, dtype=np.int64)
    
    for p in small_primes:
        # Vectorized: for each power p^k, multiply sf at indices divisible by p^k
        pk = p
        while pk <= max_val:
            # All multiples of pk get an additional factor of p
            multiples = np.arange(pk, max_val + 1, pk, dtype=np.int64)
            sf[multiples] *= p
            pk *= p
        # Check time
        if time.time() > deadline - 10:
            break
    
    # Now sf[x] gives the small-prime factor for each x
    
    # For sliding window: n in [1000, 1000000], window is [n+1, n+24]
    
    # Precompute count of 1s (rough numbers) in sliding window
    # rough[x] = 1 if sf[x] == 1 else 0
    rough = (sf == 1).astype(np.int32)
    
    window_size = 24
    min_n = 1000
    max_n = 1000000
    
    # Precompute cumulative sum of rough (vectorized)
    cum_rough = np.zeros(max_val + 2, dtype=np.int32)
    np.cumsum(rough, out=cum_rough[1:])
    
    # Compute rough counts for all valid n
    n_arr = np.arange(min_n, max_n + 1, dtype=np.int64)
    rough_counts = cum_rough[n_arr + 24] - cum_rough[n_arr]
    
    # Find the maximum rough count
    max_rough = int(np.max(rough_counts))
    
    # Get all n that achieve max_rough
    candidates = n_arr[rough_counts == max_rough]
    
    best_score = 25
    best_n = 1000
    
    if len(candidates) > 0:
        # Among these, find the one with minimum distinct small-prime factors
        for idx in range(len(candidates)):
            if time.time() > deadline:
                break
            n = int(candidates[idx])
            
            # Count distinct small-prime factors in [n+1, n+24]
            window_sf = sf[n+1:n+25]
            count = len(set(window_sf.tolist()))
            
            if count < best_score:
                best_score = count
                best_n = n
                if best_score == 1:
                    break
    
    # If no candidates found or score still high, do a broader search
    if len(candidates) == 0 or best_score > 3:
        # Fallback: search more broadly with vectorized window processing
        # Process in large chunks for efficiency
        chunk_size = 50000
        for chunk_start in range(min_n, max_n + 1, chunk_size):
            if time.time() > deadline:
                break
            
            chunk_end = min(chunk_start + chunk_size, max_n + 1)
            chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)
            
            # For each n in chunk, we need sf[n+1:n+25]
            # This is a 2D array of shape (len(chunk_n), 24)
            # We can compute distinct counts per row
            indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
            window_2d = sf[indices]  # shape (chunk_size, 24)
            
            # Count distinct values per row using a vectorized approach
            # For small distinct values, we can use: for each row, count unique
            # Use: sort each row, then count changes
            sorted_rows = np.sort(window_2d, axis=1)
            # Count distinct: 1 + number of position changes
            diffs = np.diff(sorted_rows, axis=1)
            distinct_counts = 1 + np.sum(diffs != 0, axis=1)
            
            # Find the minimum in this chunk
            min_idx = int(np.argmin(distinct_counts))
            min_count = int(distinct_counts[min_idx])
            
            if min_count < best_score:
                best_score = min_count
                best_n = int(chunk_n[min_idx])
                if best_score == 1:
                    break
    
    return best_n
# EVOLVE-BLOCK-END
