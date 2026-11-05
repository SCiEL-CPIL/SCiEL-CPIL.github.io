import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    MAX_VAL = N_MAX + WINDOW

    # Compute extracted[m] = product of p^k for p<24 where p^k || m
    # Using sieve: for each prime p, for each power p^k, multiply all multiples of p^k by p
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            extracted[p_power::p_power] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Get IDs for extracted values from index 1 to MAX_VAL
    vals = extracted[1:MAX_VAL + 1]
    unique_vals, ids = np.unique(vals, return_inverse=True)
    
    # Compute previous occurrence index for each position
    # ids is 0-indexed array of length MAX_VAL
    # We want prev[i] = index of previous occurrence of ids[i], or -1
    prev = np.full(len(ids), -1, dtype=np.int64)
    last_seen = np.full(len(unique_vals), -1, dtype=np.int64)
    
    # This loop is O(N) in Python. ~1M iterations.
    for i in range(len(ids)):
        uid = ids[i]
        prev[i] = last_seen[uid]
        last_seen[uid] = i

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # We want to count distinct elements in window n+1 to n+24
    # Which corresponds to indices n to n+23 in 'ids' (0-indexed, where ids[k] is value for k+1)
    # Window for n: numbers n+1 to n+24.
    # Indices in extracted: n+1 to n+24.
    # Indices in ids: n to n+23.
    
    # For a given index j in ids (0-based), it corresponds to number j+1.
    # It is in window for n if n <= j <= n+23.
    # So n <= j and n >= j-23.
    # It is the first occurrence in the window if prev[j] < n.
    # So n > prev[j] => n >= prev[j] + 1.
    # Range of n: [max(j-23, prev[j]+1), j]
    
    # We only care about n in [N_MIN, N_MAX]
    # Let's create a difference array for n in [0, MAX_VAL]
    diff = np.zeros(MAX_VAL + WINDOW + 2, dtype=np.int32)
    
    # Iterate over j and update diff
    # Optimization: skip if range doesn't intersect [N_MIN, N_MAX]
    for j in range(len(ids)):
        lo = j - WINDOW + 1
        if prev[j] + 1 > lo:
            lo = prev[j] + 1
        hi = j
        
        if lo > N_MAX or hi < N_MIN:
            continue
            
        # Clamp to [N_MIN, N_MAX]
        if lo < N_MIN:
            lo = N_MIN
        if hi > N_MAX:
            hi = N_MAX
            
        if lo <= hi:
            diff[lo] += 1
            diff[hi + 1] -= 1

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Compute prefix sum to get counts for each n
    counts = np.cumsum(diff)
    
    # Extract counts for n in [N_MIN, N_MAX]
    relevant_counts = counts[N_MIN:N_MAX + 1]
    
    min_count = np.min(relevant_counts)
    best_n = N_MIN + np.argmin(relevant_counts)
    
    return best_n
