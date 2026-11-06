import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to analyze n in [1000, 1000000]
    # The values involved are n+1 to n+24
    # Min value: 1000 + 1 = 1001
    # Max value: 1000000 + 24 = 1000024
    
    min_val = 1001
    max_val = 1000024
    N = max_val - min_val + 1
    
    # Initialize remaining values and the small-prime part accumulator
    rem = np.arange(min_val, max_val + 1, dtype=np.int64)
    parts = np.ones(N, dtype=np.int64)
    
    # Compute the small-prime part for each number
    for p in primes:
        mask = rem % p == 0
        while np.any(mask):
            parts[mask] *= p
            rem[mask] //= p
            # Update mask: check if the updated rem is still divisible by p
            # Note: rem elements not in mask are unchanged and were not divisible by p
            # (or became 1, which is not divisible by p > 1)
            # So we only need to check the masked indices again
            mask = rem % p == 0
            # Optimization: if no more divisible, break
            # np.any(mask) is O(N) but small constant. 
            # We can also break if mask.sum() == 0.
            if not np.any(mask):
                break
                
    # Map the smooth parts to compact integer IDs for fast distinct counting
    # The number of distinct smooth parts is small
    unique_vals, ids = np.unique(parts, return_inverse=True)
    num_unique = len(unique_vals)
    
    # Sliding window distinct count
    W = 24
    # n ranges from 1000 to 1000000
    # For a given n, the values are n+1 ... n+24
    # Index in array: val - min_val
    # n+1 -> index (n+1) - 1001 = n - 1000
    # n+24 -> index (n+24) - 1001 = n - 977
    # Window indices: [n-1000, n-977] inclusive. Size 24.
    
    # We iterate n from 1000 to 1000000
    # Let start_idx = n - 1000.
    # n=1000 -> start_idx=0. Window [0, 23].
    # n=1000000 -> start_idx=999000. Window [999000, 999023].
    # Max index in ids is N-1 = 1000024 - 1001 = 999023.
    # So the last window fits perfectly.
    
    # Initialize frequency array
    freq = np.zeros(num_unique, dtype=np.int32)
    
    # Initial window for n=1000 (indices 0 to 23)
    initial_window = ids[0:W]
    # Count frequencies
    np.add.at(freq, initial_window, 1)
    distinct_count = np.sum(freq > 0)
    
    best_n = 1000
    best_distinct = distinct_count
    
    # Slide the window
    # n goes from 1001 to 1000000
    # start_idx goes from 1 to 999000
    for n in range(1001, 1000001):
        # Check time
        if time.time() > deadline:
            break
            
        start_idx = n - 1000
        out_idx = start_idx - 1       # The element leaving the window
        in_idx = start_idx + W - 1    # The element entering the window
        
        out_id = ids[out_idx]
        in_id = ids[in_idx]
        
        # Remove out_id
        freq[out_id] -= 1
        if freq[out_id] == 0:
            distinct_count -= 1
            
        # Add in_id
        freq[in_id] += 1
        if freq[in_id] == 1:
            distinct_count += 1
            
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = n
            # If we find 1, it's the theoretical minimum (all parts identical)
            # If we find 2, it's very good.
            # We can't break on 1 easily because maybe there's no 1? 
            # But 1 is the absolute min. If we find it, we can break?
            # The task is to MINIMIZE. If we find 1, we can't do better.
            if best_distinct == 1:
                break
                
    return best_n
