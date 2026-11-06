import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    lower = 1000
    upper = 1000000
    max_val = upper + 24
    size = max_val - lower + 1  # values from lower to max_val

    # Precompute small-prime part for each number in [lower, max_val]
    # small_part[m] = product of p^v_p(m) for p in primes (primes < 24)
    small_parts = [1] * size

    for p in primes:
        first_mult = ((lower + p - 1) // p) * p
        for m in range(first_mult, max_val + 1, p):
            idx = m - lower
            temp = m
            while temp % p == 0:
                small_parts[idx] *= p
                temp //= p

    # Sliding window scan for best n
    # For n, we look at small_parts for n+1 through n+24
    # Index in small_parts: (n+1) - lower through (n+24) - lower
    best_n = lower
    best_count = 25  # max possible distinct values is 24, use 25 as initial

    # Initialize window for n = lower: indices (lower+1-lower) to (lower+24-lower) = 1 to 24
    window_counts = defaultdict(int)
    for i in range(1, 25):  # indices 1 through 24 in small_parts
        window_counts[small_parts[i]] += 1
    current_distinct = len(window_counts)
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = lower

    # Slide the window: for n from lower+1 to upper
    # When moving from n to n+1:
    #   Remove small_parts[(n+1) - lower] = small_parts[n - lower + 1]
    #   Add small_parts[(n+25) - lower] = small_parts[n - lower + 25]
    for n in range(lower + 1, upper + 1):
        if time.time() > deadline:
            break
        
        # Index of value leaving the window: (n) - lower (this was n+1-(lower+1) for previous n)
        # Actually: for previous n-1, the window was [(n-1)+1 - lower, (n-1)+24 - lower] = [n-lower, n+23-lower]
        # For current n, the window is [n+1-lower, n+24-lower]
        # So we remove small_parts[n - lower] and add small_parts[n + 24 - lower]
        out_idx = n - lower
        in_idx = n + 24 - lower
        
        out_val = small_parts[out_idx]
        in_val = small_parts[in_idx]
        
        window_counts[out_val] -= 1
        if window_counts[out_val] == 0:
            del window_counts[out_val]
        window_counts[in_val] += 1
        
        current_distinct = len(window_counts)
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
            if best_count == 1:
                break

    return best_n
