import time

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin

    lower = 1000
    upper = 1000000
    max_val = upper + 24
    nvals = max_val - lower + 1

    # Precompute small-prime part for each number in [lower, max_val]
    # Index i corresponds to number lower + i
    sp_parts = [1] * nvals

    for p in _SMALL_PRIMES:
        # First multiple of p in [lower, max_val]
        first_mult = ((lower + p - 1) // p) * p
        for m in range(first_mult, max_val + 1, p):
            idx = m - lower
            temp = m
            while temp % p == 0:
                sp_parts[idx] *= p
                temp //= p

    # Sliding window scan
    # Window for n covers [n+1, n+24]
    # Indices in sp_parts:
    # n+1 -> (n+1) - lower
    # n+24 -> (n+24) - lower
    
    best_n = lower
    best_count = float('inf')
    
    # Initialize window for n = lower
    # Range of indices: (lower + 1 - lower) to (lower + 24 - lower)
    # i.e., 1 to 24
    freq = {}
    distinct_count = 0
    
    start_idx = 1
    end_idx = 24
    
    for i in range(start_idx, end_idx + 1):
        val = sp_parts[i]
        if val not in freq:
            freq[val] = 0
            distinct_count += 1
        freq[val] += 1
        
    best_count = distinct_count
    best_n = lower

    # Slide window
    # For n from lower + 1 to upper
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
            
        # Remove element at start_idx (which is n - lower)
        # Wait, for n=lower, start_idx=1 (num lower+1).
        # For n=lower+1, the window is [lower+2, lower+25].
        # We remove lower+1 (idx 1) and add lower+25 (idx 25).
        
        # Current start_idx corresponds to number (lower + start_idx)
        # We are moving from window starting at n to n+1.
        # Old window: [n+1, n+24] -> indices [n+1-lower, n+24-lower]
        # New window: [n+2, n+25] -> indices [n+2-lower, n+25-lower]
        
        # Remove index: n + 1 - lower
        # Add index: n + 25 - lower
        
        rem_idx = n + 1 - lower
        add_idx = n + 25 - lower
        
        val_out = sp_parts[rem_idx]
        freq[val_out] -= 1
        if freq[val_out] == 0:
            del freq[val_out]
            distinct_count -= 1
            
        val_in = sp_parts[add_idx]
        if val_in not in freq:
            freq[val_in] = 0
            distinct_count += 1
        freq[val_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break # Can't get better

    return best_n
