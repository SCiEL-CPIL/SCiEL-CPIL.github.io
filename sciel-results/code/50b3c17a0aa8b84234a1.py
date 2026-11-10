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
        pk = p
        while pk <= max_val:
            np.multiply(sf[pk::pk], p, out=sf[pk::pk])
            pk *= p
        if time.time() > deadline - 20:
            break

    window_size = 24
    min_n = 1000
    max_n = 1000000

    # Full vectorized scan to find the global minimum
    # We process in chunks to manage memory and allow early termination if time is tight
    # But given the size (1M * 24), a single pass might be okay if optimized.
    # Let's try a single large pass first, but chunked to be safe with memory/time.
    
    best_score = 25
    best_n = min_n

    # Use int32 for indices to save memory if possible, but int64 is safer for n+24
    # We will compute distinct counts for all n in [min_n, max_n]
    
    # To speed up sorting, we can map sf values to smaller integers if they are sparse,
    # but sf values are products of small primes, so they are not that large.
    # Sorting 24 elements is O(24 log 24) which is small.
    
    # Let's process in chunks of 100,000 to stay well within memory limits and allow time checks
    chunk_size = 100000
    
    for chunk_start in range(min_n, max_n + 1, chunk_size):
        if time.time() > deadline - 5:
            break

        chunk_end = min(chunk_start + chunk_size, max_n + 1)
        num_in_chunk = chunk_end - chunk_start
        chunk_n = np.arange(chunk_start, chunk_end, dtype=np.int64)

        # Create indices for the window [n+1, n+24]
        # Shape: (num_in_chunk, 24)
        indices = chunk_n[:, None] + np.arange(1, 25)[None, :]
        
        # Get the small-prime factors
        window_2d = sf[indices]
        
        # Sort each row
        sorted_rows = np.sort(window_2d, axis=1)
        
        # Count distincts: 1 + number of changes
        diffs = np.diff(sorted_rows, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        
        # Find minimum in this chunk
        min_count = int(distinct_counts.min())
        if min_count < best_score:
            best_score = min_count
            best_n = int(chunk_n[distinct_counts.argmin()])
            
            # If we found a very low score, we might want to continue to ensure it's global min
            # but if we hit 2 or 1, it's likely optimal.
            if best_score <= 2:
                # We can break early if we are confident, but let's finish the current chunk
                # and then maybe break if time allows or if we are certain.
                # For now, continue to see if we can find 1.
                pass

    return best_n
# EVOLVE-BLOCK-END
