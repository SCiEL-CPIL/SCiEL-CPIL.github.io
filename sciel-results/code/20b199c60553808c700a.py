"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Range of numbers we need: n+i for n in [1000, 1000000], i in [1, 24]
    # So numbers range from 1001 to 1000024
    lo = 1001
    hi = 1000024
    size = hi - lo + 1
    
    # Compute small-prime factor for each number in [lo, hi]
    # small_factor[m] = product of p^v where p in primes and p^v divides m
    # For each prime p in primes, compute the p-part of each number
    
    # Create array of numbers
    nums = np.arange(lo, hi + 1, dtype=np.int64)
    
    # small_factor starts as 1 for all
    small_factor = np.ones(size, dtype=np.int64)
    
    for p in primes:
        # Extract the p-part of each number
        temp = nums.copy()
        for _ in range(20):  # enough for p=2
            mask = temp % p == 0
            if not np.any(mask):
                break
            temp[mask] //= p
            small_factor[mask] *= p
    
    # Now small_factor[i] is the small-prime factor of (lo + i)
    # n+i ranges from n+1 to n+24
    # Window indices: n-1000 to n-976 (length 24)
    
    # Sliding window distinct count using vectorized approach
    W = 24
    N = 1000000 - 1000 + 1  # number of valid n values = 999001
    
    # Build all windows as a 2D array: shape (N, W)
    # window[s, j] = small_factor[s + j] for s in [0, N-1], j in [0, W-1]
    # Use stride tricks for memory efficiency
    from numpy.lib.stride_tricks import as_strided
    
    # Create a view of all windows
    window_view = as_strided(small_factor, shape=(N, W), strides=(small_factor.strides[0], small_factor.strides[0]))
    
    # For each window, count distinct values
    # Vectorized approach: for each window, use np.unique or count frequencies
    # Since W=24 is small, we can use a different approach:
    # For each position in the window, check if it's the first occurrence in that window
    
    # Alternative: compute distinct count for each window
    # For window s, distinct count = number of unique values in small_factor[s:s+W]
    
    # Efficient approach: for each element, determine if it's the first occurrence
    # in its window. An element at position i in the array is the first occurrence
    # in window s if:
    # 1. s <= i < s+W (i.e., s in [i-W+1, i])
    # 2. For all j in [s, i-1], small_factor[j] != small_factor[i]
    
    # This is complex to vectorize. Let's use a simpler approach:
    # Since W=24 is small, we can compute distinct counts using broadcasting
    # but that would be O(N*W^2) which is ~576M operations - too slow.
    
    # Better: use the fact that we only need to find the minimum.
    # Let's use a hybrid: vectorized for most, with early termination.
    
    # Actually, let's just use the sliding window with numpy optimizations.
    # The key optimization: avoid Python loop overhead by processing in chunks.
    
    # Precompute: for each position, what's the next occurrence of the same value?
    # This helps with distinct counting.
    
    # Simpler approach: just do the sliding window but with numpy for the comparisons
    best_n = 1000
    best_distinct = W  # maximum possible
    
    # Use a dict for frequency tracking - this is the most practical approach
    # for this problem size. Optimize by using local variables.
    
    freq = {}
    distinct_count = 0
    
    # First window
    for i in range(W):
        val = small_factor[i]
        freq[val] = freq.get(val, 0) + 1
        if freq[val] == 1:
            distinct_count += 1
    
    if distinct_count < best_distinct:
        best_distinct = distinct_count
        best_n = 1000
    
    # Slide window with time check
    sf = small_factor  # local reference for speed
    W_local = W
    N_local = N
    deadline_local = deadline
    start_local = start_time
    
    for s in range(1, N_local):
        old_val = sf[s - 1]
        cnt = freq[old_val] - 1
        if cnt == 0:
            del freq[old_val]
            distinct_count -= 1
        else:
            freq[old_val] = cnt
        
        new_val = sf[s + W_local - 1]
        cnt = freq.get(new_val, 0) + 1
        freq[new_val] = cnt
        if cnt == 1:
            distinct_count += 1
        
        if distinct_count < best_distinct:
            best_distinct = distinct_count
            best_n = 1000 + s
            if best_distinct == 1:
                break
        
        if s % 50000 == 0:
            if time.time() > deadline_local:
                break
    
    return best_n
# EVOLVE-BLOCK-END
