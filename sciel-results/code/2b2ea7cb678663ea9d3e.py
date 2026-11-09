def construct():
    import time
    t0 = time.time()
    DL = 85.0

    N = 123457

    # Compute largest prime factor for every integer in [1, N] using a sieve.
    # lpf[x] holds the largest prime factor of x after the sieve completes.
    lpf = [1] * (N + 1)
    for p in range(2, N + 1):
        if lpf[p] == 1:
            for m in range(p, N + 1, p):
                lpf[m] = p

    best_val = float('inf')
    best_pair = (1, N - 1)

    # Scan all splits a + b = N, a in [1, N-1].
    # Because a and b are symmetric, scanning half is sufficient, but a full scan
    # is still O(N) and trivially fast.  We break early if we hit the absolute
    # lower bound (both 1-smooth, i.e. both powers of 2) or if time runs out.
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        cur = pa if pa >= pb else pb
        if cur < best_val:
            best_val = cur
            best_pair = (a, b)
            if cur == 1:
                break
        if time.time() - t0 > DL:
            break

    return [best_pair[0], best_pair[1]]
