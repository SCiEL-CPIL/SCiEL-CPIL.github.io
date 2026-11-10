def construct():
    import time
    start = time.time()
    DEADLINE = 88.0
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
        order = sorted(range(n), key=lambda i: cands[i].bit_count())
        
        # Memoization on (k, used) to speed up repeated subproblems
        memo = {}
        def rec(k, used):
            if k == n:
                return True
            key = (k, used)
            if key in memo:
                return memo[key]
            i = order[k]
            mask = cands[i] & ~used
            result = False
            while mask:
                lsb = mask & (-mask)
                if rec(k+1, used | lsb):
                    result = True
                    break
                mask &= mask - 1
            memo[key] = result
            return result
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
    
    # First pass: evaluate all m in the first period to find candidates
    for m in range(PERIOD):
        if time.time() - start > DEADLINE:
            break
        fv = f_of(m)
        if fv > best_f:
            best_f = fv
            best_m = m
    
    # Second pass: if we have time, do a local search around the best m
    # to see if nearby values (in subsequent periods) give better results
    if time.time() - start < DEADLINE - 5.0:
        # Try m values in the next few periods around the best_m
        for offset in range(1, 5):
            if time.time() - start > DEADLINE:
                break
            for delta in range(-50, 51):
                m2 = (best_m + offset * PERIOD + delta) % PERIOD
                if m2 == best_m:
                    continue
                fv = f_of(m2)
                if fv > best_f:
                    best_f = fv
                    best_m = m2
    
    return int(best_m)
