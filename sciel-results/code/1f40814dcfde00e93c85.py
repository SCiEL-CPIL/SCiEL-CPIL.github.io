import time
import numpy as np
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 60s limit

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    # sp[i] = product of p^a for p in _SMALL_PRIMES where a = v_p(lower+i)
    sp_parts = np.ones(nvals, dtype=np.int64)
    
    for p in _SMALL_PRIMES:
        # Find indices where (lower + idx) is divisible by p
        # (lower + idx) % p == 0 => idx % p == (p - lower % p) % p
        first_idx = (p - lower % p) % p
        # Indices: first_idx, first_idx + p, first_idx + 2p, ...
        idxs = np.arange(first_idx, nvals, p)
        if len(idxs) == 0:
            continue
        
        # For each such index, compute p^(v_p(lower + idx))
        # We need to repeatedly divide by p
        vals = (lower + idxs).astype(np.int64)
        mult = np.ones(len(idxs), dtype=np.int64)
        
        while True:
            mask = (vals % p == 0)
            if not mask.any():
                break
            vals[mask] //= p
            mult[mask] *= p
        
        sp_parts[idxs] *= mult

    # Convert to list for faster access in sliding window
    sp_list = sp_parts.tolist()

    # Sliding window over 24 consecutive values
    # Window for n: values n+1, n+2, ..., n+24
    # In sp_list, index for value v is v - lower
    # So n+1 -> index (n+1) - lower, n+24 -> index (n+24) - lower

    best_n = lower
    best_count = float('inf')

    # Initialize window for n = lower: values lower+1 to lower+24
    # indices: 1, 2, ..., 24
    window_counts = defaultdict(int)
    for i in range(1, 25):
        window_counts[sp_list[i]] += 1
    current_distinct = len(window_counts)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower

    # Slide the window
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break

        # Remove the value that's leaving: n (index n - lower)
        # Add the value that's entering: n+24 (index n+24 - lower)
        out_val = sp_list[n - lower]
        in_val = sp_list[n + 24 - lower]

        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]
        window_counts[in_val] += 1

        current_distinct = len(window_counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count <= 1:
                break  # Can't do better than 1

    return best_n
