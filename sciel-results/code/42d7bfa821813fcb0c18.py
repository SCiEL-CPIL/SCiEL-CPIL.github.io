"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from math import isqrt
    import time

    start_time = time.time()
    TIME_LIMIT = 110  # Leave 10s margin

    N_MAX = 5_000_000

    # Generate powerful numbers up to N_MAX
    pw_set = set()

    # Precompute squares of small numbers
    b = 1
    while b * b * b <= N_MAX:
        # Check if b is square-free
        is_square_free = True
        d = 2
        temp_b = b
        while d * d <= temp_b:
            if temp_b % (d * d) == 0:
                is_square_free = False
                break
            d += 1
        if is_square_free:
            b3 = b * b * b
            max_a = isqrt(N_MAX // b3)
            for a in range(1, max_a + 1):
                n = a * a * b3
                if n <= N_MAX:
                    pw_set.add(n)
        b += 1

    pw = sorted(pw_set)
    pw_len = len(pw)

    # Count sums using optimized approach
    from collections import defaultdict
    counts = defaultdict(int)

    # Local variable for speed
    pw_list = pw
    counts_local = counts
    n_max = N_MAX

    for i in range(pw_len):
        a = pw_list[i]
        if time.time() - start_time > TIME_LIMIT:
            break
        # Binary search for max j such that pw_list[j] <= n_max - a
        # Use bisect
        import bisect
        max_j = bisect.bisect_right(pw_list, n_max - a)
        for j in range(i, max_j):
            s = a + pw_list[j]
            counts_local[s] += 1

    if not counts:
        return {'N': 2, 'pairs': [[1, 1]]}

    best_N, best_count = max(counts.items(), key=lambda x: x[1])

    # Retrieve pairs
    pairs = []
    pw_set_local = pw_set
    for a in pw:
        if 2 * a > best_N:
            break
        b_val = best_N - a
        if b_val in pw_set_local:
            pairs.append([a, b_val])

    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
