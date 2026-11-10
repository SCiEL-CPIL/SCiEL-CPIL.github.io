def construct():
    import time
    start = time.time()
    DEADLINE = 85.0
    n = 12
    PERIOD = 27720  # lcm(1..12)
    MAXL = 144
    MINL = 12

    def can(m, L):
        cands = []
        for i in range(1, n+1):
            r = (-(m+1)) % i
            mask = 0
            for p in range(r, L, i):
                mask |= 1 << p
            cands.append(mask)
        # Precompute union of all candidate masks for Hall's condition
        all_union = 0
        for cm in cands:
            all_union |= cm
        # Order by candidate count (ascending) for better pruning
        order = sorted(range(n), key=lambda i: cands[i].bit_count())
        # Memoization key: (k, used_mask) -> result
        memo = {}
        def rec(k, used):
            key = (k, used)
            if key in memo:
                return memo[key]
            if k == n:
                memo[key] = True
                return True
            i = order[k]
            mask = cands[i] & ~used
            if mask == 0:
                memo[key] = False
                return False
            # Hall's condition: remaining candidates must be sufficient
            # For positions k..n-1, the union of their candidates minus used must have enough bits
            remaining_union = 0
            for j in range(k, n):
                remaining_union |= cands[order[j]]
            remaining_union &= ~used
            if remaining_union.bit_count() < (n - k):
                memo[key] = False
                return False
            while mask:
                lsb = mask & (-mask)
                if rec(k+1, used | lsb):
                    memo[key] = True
                    return True
                mask &= mask - 1
            memo[key] = False
            return False
        return rec(0, 0)

    def f_of(m):
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
    # First pass: scan all residues in the period
    for m in range(PERIOD):
        if time.time() - start > DEADLINE:
            break
        fv = f_of(m)
        if fv > best_f:
            best_f = fv
            best_m = m
    # Second pass: if time remains, do local refinement around best_m
    # Check neighbors of best_m within the period (wrap around)
    if time.time() - start < DEADLINE - 5:
        for delta in range(1, 50):
            if time.time() - start > DEADLINE:
                break
            for dm in [best_m - delta, best_m + delta]:
                dm_mod = dm % PERIOD
                fv = f_of(dm_mod)
                if fv > best_f:
                    best_f = fv
                    best_m = dm_mod
    return int(best_m)
