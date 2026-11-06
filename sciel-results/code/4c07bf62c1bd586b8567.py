"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find n in [1000, 1000000] such that for each n+i (1<=i<=24),
    we extract the smooth part (product of prime powers for primes < 24),
    and minimize the number of DISTINCT smooth parts across all 24 values.
    
    Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23
    """
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    max_val = 1000000 + 24
    W = 24

    # Precompute smooth part for each number
    # smooth[x] = product of p^v_p(x) for all primes p < 24
    smooth = np.ones(max_val + 1, dtype=np.int32)

    for p in primes:
        for multiple in range(p, max_val + 1, p):
            power = 1
            temp = multiple
            while temp % p == 0:
                power *= p
                temp //= p
            smooth[multiple] *= power

    # Sliding window: for n in [1000, 1000000], look at smooth[n+1]..smooth[n+24]
    # Count distinct values using frequency dictionary
    best_n = 1000
    best_count = W  # worst case: all 24 distinct

    # Initialize frequency dictionary for first window (n=1000: indices 1001..1024)
    freq = {}
    sp = smooth  # local reference for speed

    for i in range(1001, 1001 + W):
        val = sp[i]
        freq[val] = freq.get(val, 0) + 1

    distinct = len(freq)
    best_n = 1000
    best_count = distinct

    # Slide the window
    check_interval = 100000
    next_check = check_interval
    n_end = 1000000

    for n in range(1001, n_end + 1):
        # Remove element at position n (leftmost of previous window)
        val_out = sp[n]
        cnt = freq[val_out] - 1
        if cnt == 0:
            del freq[val_out]
        else:
            freq[val_out] = cnt

        # Add element at position n+W (new rightmost)
        val_in = sp[n + W]
        freq[val_in] = freq.get(val_in, 0) + 1

        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count <= 2:
                break

        if n >= next_check:
            if time.time() > deadline:
                break
            next_check += check_interval

    return best_n
# EVOLVE-BLOCK-END
