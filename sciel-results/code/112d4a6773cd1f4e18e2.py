import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 90s
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000025
    
    # Precompute small part for each number 1..max_val
    small_part = np.arange(max_val, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk < max_val:
            for j in range(pk, max_val, pk):
                small_part[j] //= p
            pk *= p
    
    if time.time() > deadline:
        return 1000
    
    min_n = 1000
    max_n = 1000000
    n_count = max_n - min_n + 1  # 999001
    
    # Fully vectorized batched scan: compute distinct counts for ALL n at once
    # in large batches, using numpy sort + diff to count distinct values.
    # This eliminates the Python loop entirely.
    
    best_count = 25
    best_n = min_n
    
    # Process in batches to manage memory
    batch_size = 50000
    offsets = np.arange(1, 25)  # shape (24,)
    
    for batch_start in range(min_n, max_n + 1, batch_size):
        if time.time() > deadline - 3:
            break
        
        batch_end = min(batch_start + batch_size, max_n + 1)
        batch_n = np.arange(batch_start, batch_end)
        batch_count = batch_end - batch_start
        
        # Build indices: shape (batch_count, 24)
        # For n in batch, we need small_part[n+1 : n+25]
        indices = batch_n[:, None] + offsets[None, :]  # (batch_count, 24)
        vals = small_part[indices]  # (batch_count, 24)
        
        # Sort each row to count distinct values
        sorted_vals = np.sort(vals, axis=1)
        changes = np.diff(sorted_vals, axis=1) != 0
        distinct_counts = 1 + changes.sum(axis=1)
        
        # Find the minimum in this batch
        local_min_idx = np.argmin(distinct_counts)
        local_min_count = int(distinct_counts[local_min_idx])
        
        if local_min_count < best_count:
            best_count = local_min_count
            best_n = int(batch_n[local_min_idx])
            if best_count == 1:
                return best_n
    
    return int(best_n)
