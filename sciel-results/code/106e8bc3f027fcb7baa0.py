def construct():
    import time
    import numpy as np
    start = time.time()
    DEADLINE = 82.0
    n = 12
    PERIOD = 27720  # lcm(1..12)
    MAXL = 144
    MINL = 12

    # Precompute for each i, the residues mod i
    # For a given m, position p (0-indexed in window) is valid for i if (m+1+p) % i == 0
    # i.e., p % i == (-m-1) % i

    def can_match(m, L):
        # For each i in 1..12, find which positions 0..L-1 are valid
        # Position p is valid for i if (m + 1 + p) % i == 0
        # So p ≡ (-m-1) % i (mod i)
        # Candidates for i: p in [0, L) with p ≡ r_i (mod i), where r_i = (-m-1) % i

        # Build candidate lists
        cands = []
        for i in range(1, n + 1):
            r = (-m - 1) % i
            # positions: r, r+i, r+2i, ... < L
            if r >= L:
                # no candidates
                return False
            cnt = (L - 1 - r) // i + 1
            if cnt < 1:
                return False
            cands.append((i, r, cnt))

        # Sort by scarcity (fewest candidates first) for better pruning
        order = sorted(range(n), key=lambda idx: cands[idx][2])

        # Quick Hall's condition: for the k most constrained, check if union of candidates >= k
        # This is a necessary condition (not sufficient), but catches easy failures
        # For now, just do DFS with good ordering

        # Use iterative DFS with bitmasks
        # Positions 0..L-1, L <= 144, so we need up to 144 bits
        # Python ints handle arbitrary precision

        # Precompute candidate bitmasks
        masks = []
        for idx in order:
            i, r, cnt = cands[idx]
            mask = 0
            p = r
            while p < L:
                mask |= (1 << p)
                p += i
            masks.append(mask)

        # DFS with memo on (k, used) — but used can be huge, so use simple recursion
        # With good ordering and early termination, this should be fast

        # Use a stack-based approach to avoid recursion depth issues
        # Actually for n=12, recursion depth is at most 12, so recursion is fine

        def dfs(k, used):
            if k == n:
                return True
            mask = masks[k] & ~used
            while mask:
                lsb = mask & (-mask)
                if dfs(k + 1, used | lsb):
                    return True
                mask &= mask - 1
            return False

        return dfs(0, 0)

    def f_of_m(m):
        # Binary search for the smallest L in [MINL, MAXL] such that can_match(m, L)
        if not can_match(m, MAXL):
            return -1
        lo, hi = MINL, MAXL
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    # Search over one period
    best_m = 0
    best_f = -1

    # First pass: find the maximum f value achievable
    # Since f(m) is periodic, search m in [0, PERIOD)
    for m in range(PERIOD):
        if time.time() - start > DEADLINE:
            break
        fv = f_of_m(m)
        if fv > best_f:
            best_f = fv
            best_m = m
        # If we hit the theoretical max, we can stop early
        if best_f >= MAXL:
            break

    return int(best_m % PERIOD)
