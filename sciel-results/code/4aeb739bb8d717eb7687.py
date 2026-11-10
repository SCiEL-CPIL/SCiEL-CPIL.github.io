import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 90.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Initialize small_part array using NumPy for vectorized operations
    small_part = np.ones(size, dtype=np.int64)
    
    # Sieve-like computation (vectorized)
    for p in primes:
        power = p
        while power <= max_val:
            # Multiply small_part[j] by p for all j divisible by power
            # This effectively accumulates the p-part of j
            small_part[power::power] *= p
            power *= p
            
    # We need n in [1000, 1000000]
    # Window is n+1 to n+24
    # Indices in small_part: 1001 to 1000024
    # Slice: small_part[1001 : 1000025]
    
    slice_start = 1001
    slice_end = 1000025
    sp_slice = small_part[slice_start:slice_end]
    
    # Create a view of sliding windows of size 24
    # Shape: (num_windows, 24)
    num_windows = len(sp_slice) - 24 + 1
    if num_windows <= 0:
        return 1000
        
    # Use as_strided to create the window view without copying data
    # sp_slice.strides[0] is the byte size of one element
    strides = (sp_slice.strides[0], sp_slice.strides[0])
    shape = (num_windows, 24)
    
    try:
        window_view = np.lib.stride_tricks.as_strided(sp_slice, shape=shape, strides=strides)
    except Exception:
        # Fallback to standard loop if as_strided fails (unlikely)
        best_n = 1000
        best_count = 25
        for n in range(1000, max_n + 1):
            window = small_part[n+1 : n+25]
            count = len(set(window))
            if count < best_count:
                best_count = count
                best_n = n
                if best_count == 2:
                    break
            if (n - 1000) % 10000 == 0:
                if time.time() - start_time > timeout - 5.0:
                    break
        return best_n

    # Sort each window to count distinct elements
    # np.sort on a 2D array sorts along the last axis by default
    sorted_windows = np.sort(window_view, axis=1)
    
    # Calculate differences between adjacent elements in sorted windows
    diffs = np.diff(sorted_windows, axis=1)
    
    # Count distinct elements: 1 + number of non-zero differences
    counts = 1 + np.sum(diffs != 0, axis=1)
    
    # Find the index with the minimum count
    min_idx = np.argmin(counts)
    best_n = 1000 + min_idx
    
    return best_n
# EVOLVE-BLOCK-END
