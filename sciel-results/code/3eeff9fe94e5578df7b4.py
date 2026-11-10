def construct():
    import time
    start = time.time()
    DEADLINE = 85.0
    n = 12
    PERIOD = 27720  # lcm(1..12)
    MAXL = 144
    MINL = 12
    BIT_VALS = [1 << p for p in range(MAXL)]

    def can(m, L):
        """Check if perfect matching exists using bitmask DP with memoization."""
        cands = [0] * n
        for i in range(1, n+1):
            r = (-(m+1)) % i
            mask = 0
            for p in range(r, L, i):
                mask |= BIT_VALS[p]
            cands[i-1] = mask

        # Quick check
        for mask in cands:
            if mask == 0:
                return False

        # Sort indices by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda i: cands[i].bit_count())

        memo = {}
        def rec(used):
            k = used.bit_count()
            if k == n:
                return True
            if used in memo:
                return memo[used]
            i = order[k]
            mask = cands[i] & ~used
            result = False
            while mask:
                lsb = mask & (-mask)
                if rec(used | lsb):
                    result = True
                    break
                mask &= mask - 1
            memo[used] = result
            return result
        return rec(0)

    def f_of(m):
        """Find minimum L such that can(m, L) is True."""
        lo, hi = MINL, MAXL
        if not can(m, hi):
            return -1
        while lo < hi:
            mid = (lo+hi)//2
            if can(m, mid):
                hi = mid
            else:
                lo = mid+1
        return lo

    best_m = 0
    best_f = -1

    # First pass: evaluate all m in the first period
    for m in range(PERIOD):
        if time.time() - start > DEADLINE:
            break
        fv = f_of(m)
        if fv > best_f:
            best_f = fv
            best_m = m
            if best_f == MAXL:
                return int(best_m)

    # Second pass: local search around best_m in subsequent periods
    if time.time() - start < DEADLINE - 5.0:
        for offset in range(1, 5):
            if time.time() - start > DEADLINE:
                break
            for delta in range(-100, 101):
                m2 = (best_m + offset * PERIOD + delta) % PERIOD
                if m2 == best_m:
                    continue
                fv = f_of(m2)
                if fv > best_f:
                    best_f = fv
                    best_m = m2

    # Third pass: broader sampling if time remains
    if time.time() - start < DEADLINE - 10.0:
        import random
        random.seed(12345)
        for _ in range(2000):
            if time.time() - start > DEADLINE:
                break
            m2 = random.randint(0, 1000000)
            fv = f_of(m2)
            if fv > best_f:
                best_f = fv
                best_m = m2

    return int(best_m)
