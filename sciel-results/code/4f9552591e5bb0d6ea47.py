import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need lp[j] for j in [1001, 1000024]
    # lp[j] = j with all factors of primes in `primes` removed
    # i.e., lp[j] = j / (largest d|j such that all prime factors of d are in primes)
    
    # Precompute for j in [1, 1000024]
    max_j = 1000024
    lp = np.arange(max_j + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_j:
            for j in range(pk, max_j + 1, pk):
                lp[j] //= p
            pk *= p
    
    if time.time() > deadline:
        return 1000
    
    # lp[j] is the large part. We want windows n in [1000, 1000000]
    # Window: lp[n+1], ..., lp[n+24]
    # Indices: n+1 to n+24, so n+1 in [1001, 1000024], n+24 in [1024, 1000024]
    # n in [1000, 1000000]
    
    # Strategy: find windows with minimal distinct count
    # First, identify "easy" windows where all lp values are 1
    # Then windows with few distinct small values
    
    # Let's compute for each starting index s = n+1 (s in [1001, 1000001]),
    # the 24 values lp[s..s+23]
    
    # To find windows with all lp == 1:
    # lp[j] == 1 means j is 23-smooth
    # Find runs of consecutive 23-smooth numbers of length >= 24
    
    is_smooth = (lp == 1)
    
    # Find runs of True in is_smooth
    # A window starting at s is all-1 if is_smooth[s..s+23] are all True
    
    # Use cumulative sum or just check
    # For each s, check if min(is_smooth[s:s+24]) == 1
    # This is a sliding window min of 24 on a boolean array
    
    # Convert to int
    smooth_int = is_smooth.astype(np.int8)
    
    # Sliding window sum: if sum of 24 consecutive == 24, all are 1
    # Use cumsum
    cumsum = np.cumsum(smooth_int)
    # sum[s:s+24] = cumsum[s+24-1] - (cumsum[s-1] if s>0 else 0)
    
    s_min = 1001
    s_max = 1000001  # n=1000000 -> s=1000001, s+23=1000024
    
    # cumsum is 0-indexed, cumsum[i] = sum of smooth_int[0..i]
    # sum of smooth_int[s..s+23] = cumsum[s+23] - (cumsum[s-1] if s>=1 else 0)
    
    # Let's just compute for all s in [s_min, s_max]
    # s+23 ranges from s_min+23 to s_max+23 = 1000024
    # cumsum index: s+23, and s-1
    
    # To avoid boundary issues, let's just loop over candidate s values
    # But first, let's find all s where the window might be good
    
    # Simpler: just iterate over all n and compute distinct count, but only for promising windows
    # Promising: max(lp in window) is small
    
    # Let's first find windows where max lp <= some threshold
    # Compute sliding window max of lp
    
    # Actually, let's just directly compute distinct count for all windows
    # 10^6 windows, each 24 elements. In Python this might take a few seconds.
    
    # Let's use a vectorized approach:
    # For each possible value v, create a boolean array where lp[j] == v
    # Then for each window, count how many distinct v's appear
    
    # This is complex. Let's just do a direct but optimized loop.
    
    best_n = 1000
    best_distinct = 100
    
    # Pre-extract the relevant slice
    # lp[1001:1000025] gives us lp[1001] to lp[1000024]
    # For n, we need lp[n+1:n+25]
    
    # Let's first try to find all-1 windows
    # is_smooth[s:s+24].all() for s in [1001, 1000001]
    
    # Use numpy: create array of window sums
    # window_sum[s] = sum of is_smooth[s:s+24]
    # If window_sum[s] == 24, all are smooth
    
    # Compute using cumsum
    # cumsum[i] = sum of is_smooth[0..i]
    # sum of is_smooth[s:s+24] = cumsum[s+23] - (cumsum[s-1] if s > 0 else 0)
    
    # Let's create an array for s in [s_min, s_max]
    # s ranges from 1001 to 1000001
    # s+23 ranges from 1024 to 1000024
    # s-1 ranges from 1000 to 1000000
    
    cumsum_full = np.cumsum(smooth_int)
    # cumsum_full[i] = sum of smooth_int[0..i]
    
    s_arr = np.arange(s_min, s_max + 1)
    # sum of smooth_int[s:s+24] for each s
    # = cumsum_full[s+23] - cumsum_full[s-1]
    # cumsum_full[s+23] where s+23 in [1024, 1000024]
    # cumsum_full[s-1] where s-1 in [1000, 1000000]
    
    window_sum = cumsum_full[s_arr + 23] - cumsum_full[s_arr - 1]
    
    # Find s where window_sum == 24
    all_smooth_mask = (window_sum == 24)
    if np.any(all_smooth_mask):
        # Found a window with all lp == 1, distinct count = 1
        first_s = s_arr[np.argmax(all_smooth_mask)]
        return int(first_s - 1)  # n = s - 1
    
    if time.time() > deadline:
        return 1000
    
    # No all-1 window. Now find windows with minimal distinct count.
    # Let's focus on windows where max lp is small.
    # Compute sliding window max of lp for window size 24
    
    # For each s, max(lp[s:s+24])
    # Use a simple approach: for each offset k in 0..23, take lp[s+k], then max over k
    
    # Let's create a 2D view: for each s, the 24 values are lp[s:s+24]
    # This is a sliding window. We can use stride tricks or just loop over 24 offsets.
    
    # max_lp[s] = max(lp[s], lp[s+1], ..., lp[s+23])
    
    # Initialize with lp[s]
    max_lp = np.empty(len(s_arr), dtype=np.int64)
    max_lp[:] = lp[s_arr]
    
    for k in range(1, 24):
        np.maximum(max_lp, lp[s_arr + k], out=max_lp)
    
    # Now sort by max_lp, take top candidates
    # Windows with small max_lp are more likely to have few distinct values
    
    # Let's take the top 50000 windows by smallest max_lp
    top_k = 50000
    if len(max_lp) > top_k:
        part_idx = np.argpartition(max_lp, top_k)[:top_k]
    else:
        part_idx = np.arange(len(max_lp))
    
    # Sort these by max_lp for better ordering
    part_idx = part_idx[np.argsort(max_lp[part_idx])]
    
    # Now evaluate distinct count for these candidates
    for idx in part_idx:
        if time.time() > deadline:
            break
        s = s_arr[idx]
        vals = lp[s:s+24]
        distinct = len(np.unique(vals))
        if distinct < best_distinct:
            best_distinct = distinct
            best_n = s - 1
            if best_distinct == 2:
                # Can't get better than 2 without all-1
                # But let's continue to see if we can find 1... we already checked
                break
    
    return int(best_n)
