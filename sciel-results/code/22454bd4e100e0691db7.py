import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23], dtype=np.int64)
    
    # We want to find n in [1000, 1000000] such that the number of distinct
    # primes p < 24 dividing any of n+1, ..., n+24 is minimized.
    
    # The range of n is [1000, 1000000].
    # The numbers involved are n+1 to n+24, so up to 1000024.
    # We can process the entire range [1000, 1000000] for n.
    # Let's define the array of n values.
    n_min = 1000
    n_max = 1000000
    
    # To make memory and computation manageable, we can process in chunks, 
    # but given the size (1M * 24 * 9 operations), it's actually very small.
    # 1M * 24 = 24M numbers. 
    # Checking divisibility by 9 primes: 24M * 9 = 216M operations.
    # NumPy handles this in < 1 second usually.
    
    # However, we need to be careful with memory.
    # Let's process in chunks of 100,000 candidates for n.
    
    best_n = n_min
    best_score = float('inf')
    
    chunk_size = 100000
    
    # We need to check n in [n_min, n_max].
    # For a given chunk of n values, we create a 2D array of shape (chunk_size, 24)
    # containing the values n+i.
    # Then we check divisibility by each prime.
    
    for start in range(n_min, n_max + 1, chunk_size):
        if time.time() > deadline:
            break
            
        end = min(start + chunk_size, n_max + 1)
        n_vals = np.arange(start, end, dtype=np.int64)
        
        # Create array of n+i for i=1..24
        # Shape: (num_n, 24)
        # n_vals[:, None] + np.arange(1, 25)
        nums = n_vals[:, None] + np.arange(1, 25, dtype=np.int64)[None, :]
        
        # For each prime, create a boolean mask of where it divides nums
        # We want to know, for each n (row), which primes appear in that row.
        
        # distinct_primes_count = np.zeros(len(n_vals), dtype=np.int32)
        
        # We can accumulate a bitmask or just count distinct primes per row.
        # Let's use a set of primes present per row? No, vectorize.
        
        # For each prime p:
        # mask = (nums % p == 0)  # Shape (num_n, 24)
        # row_has_p = mask.any(axis=1)  # Shape (num_n,)
        # distinct_primes_count += row_has_p.astype(np.int32)
        
        # This is 9 iterations of 100k * 24 operations. Very fast.
        
        distinct_count = np.zeros(len(n_vals), dtype=np.int32)
        
        for p in primes:
            # Check if any number in the window is divisible by p
            # Use modulo
            mask = (nums % p == 0)
            row_has_p = mask.any(axis=1)
            distinct_count += row_has_p.astype(np.int32)
        
        # Find the minimum in this chunk
        min_val = np.min(distinct_count)
        if min_val < best_score:
            best_score = min_val
            # Find the first index with this min value
            idx = np.where(distinct_count == best_score)[0][0]
            best_n = n_vals[idx]
            
            # If best_score is very low (e.g., 3 or 4), we might stop early?
            # The theoretical minimum is likely around 4 or 5.
            # If we find 3, it's extremely rare.
            # Let's continue to be sure, but we can break if we find 2 (impossible) or 3 (very hard).
            # Actually, let's just run the whole thing as it's fast.

    return int(best_n)
