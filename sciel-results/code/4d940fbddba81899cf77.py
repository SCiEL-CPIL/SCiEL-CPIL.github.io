def construct():
    import time
    import numpy as np

    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    lo = 1001
    hi = 1000024  # inclusive
    size = hi - lo + 1  # 1000024

    smooth_part = np.ones(size, dtype=np.int64)
    
    for p in small_primes:
        power = p
        while power <= hi:
            first = ((lo + power - 1) // power) * power
            if first <= hi:
                indices = (np.arange(first, hi + 1, power) - lo)
                smooth_part[indices] *= p
            power *= p
            if power > hi:
                break
            if time.time() > deadline - 20:
                break

    if time.time() > deadline - 20:
        return 1000

    unique_smooth, inverse = np.unique(smooth_part, return_inverse=True)
    ids = inverse.astype(np.int32)
    
    num_windows = 1000000 - 1000 + 1  # 1000001

    distinct_counts = np.ones(num_windows, dtype=np.int32)
    
    for j in range(1, 24):
        elem = ids[j:j+num_windows]
        is_new = np.ones(num_windows, dtype=bool)
        for k in range(j):
            prev = ids[k:k+num_windows]
            is_new &= (elem != prev)
        distinct_counts += is_new.astype(np.int32)
        
        if time.time() > deadline - 5:
            break

    min_count = np.min(distinct_counts)
    min_indices = np.where(distinct_counts == min_count)[0]
    best_n = 1000 + min_indices[0]
    
    return int(best_n)
