def construct():
    p = 1009
    n = 1008

    # Precompute modular inverses
    inv = [0] * (n + 1)
    for x in range(1, n + 1):
        inv[x] = pow(x, p - 2, p)

    # For each x, compute lo[x] = min(x, inv[x]) and hi[x] = max(x, inv[x])
    lo = [0] * (n + 1)
    hi = [0] * (n + 1)
    for x in range(1, n + 1):
        ix = inv[x]
        lo[x] = min(x, ix)
        hi[x] = max(x, ix)

    # For a start position s, the interval [s, e] is valid iff:
    #   for all x in [1, n], NOT (lo[x] >= s AND hi[x] <= e)
    # Equivalently, for all x with lo[x] >= s, we need hi[x] > e
    # So max valid e = min(hi[x] for x with lo[x] >= s) - 1

    # Build suffix minimum of hi over the lo dimension
    # suff_min[L] = min(hi[x] for x with lo[x] >= L)
    suff_min = [float('inf')] * (n + 2)
    current_min = float('inf')
    for L in range(n, 0, -1):
        for x in range(1, n + 1):
            if lo[x] == L:
                if hi[x] < current_min:
                    current_min = hi[x]
        suff_min[L] = current_min

    # Now find the best (s, e) pair
    best_start = 1
    best_length = 0
    for s in range(1, n + 1):
        max_e = suff_min[s] - 1
        if max_e > n:
            max_e = n
        if max_e >= s:
            length = max_e - s + 1
            if length > best_length:
                best_length = length
                best_start = s

    # Verify the result
    s = best_start
    e = s + best_length - 1
    for x in range(s, e + 1):
        ix = inv[x]
        if s <= ix <= e:
            # Fallback to a safe answer
            return [2, 1]

    return [best_start, best_length]
