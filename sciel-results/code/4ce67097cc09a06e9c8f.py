import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 85  # safety margin
    
    N = 1000000
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW  # 1000024
    
    # Compute sp[m] = product of p^v_p(m) for primes p < 24
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        # Compute exponent of p in each number
        exp = np.zeros(MAX_VAL + 1, dtype=np.int32)
        max_k = MAX_VAL // p
        for k in range(1, max_k + 1):
            exp[k * p] = 1 + exp[k]
        # Multiply sp[m] by p^exp[m]
        # Only where exp[m] > 0
        mask = exp > 0
        if np.any(mask):
            sp[mask] *= np.power(p, exp[mask], dtype=np.int64)
    
    if time.time() > deadline - 10:
        return 1000
    
    # Now find n that minimizes distinct count in window
    # sp[n+1], ..., sp[n+24]
    # For efficiency, process in chunks
    
    best_n = 1000
    best_distinct = WINDOW + 1
    
    # Use a sliding window approach
    # For each position n, the window is sp[n+1:n+25]
    # Distinct count = number of unique values in that window
    
    # Process all n at once using numpy
    # Create indices: for n from 1000 to N, window starts at n+1
    # This is a strided view
    
    from numpy.lib.stride_tricks import as_strided
    
    # The full array of windows: shape (N - 1000 + 1, 24)
    # Row i corresponds to n = 1000 + i, window is sp[1001+i : 1025+i]
    
    base = sp[1001 : N + WINDOW + 1]  # length N + WINDOW - 1000
    # Row i is base[i : i+24]
    
    # Use stride tricks to create the 2D array without copying
    row_stride = base.strides[0]
    mat = as_strided(base, shape=(N - 1000 + 1, WINDOW), strides=(row_stride, row_stride))
    
    # Count distinct in each row
    # Sort each row and count changes
    sorted_mat = np.sort(mat, axis=1)
    diffs = np.diff(sorted_mat, axis=1)
    distinct_counts = (diffs != 0).sum(axis=1) + 1
    
    # Find the minimum
    min_idx = np.argmin(distinct_counts)
    best_distinct = int(distinct_counts[min_idx])
    best_n = int(1000 + min_idx)
    
    return best_n
