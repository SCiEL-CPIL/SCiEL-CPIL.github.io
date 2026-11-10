import time

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    n = 12
    PERIOD = 27720  # lcm(1..12)
    MAXL = 144
    MINL = 12
    
    def can(m, L):
        """Check if there's a perfect matching for indices 1..n in (m, m+L] using bitmasks."""
        adj_masks = [0] * n
        
        # For each index i (1..n), build bitmask of valid positions in (m, m+L]
        for i in range(1, n + 1):
            # First multiple of i that is > m
            first = ((m + i) // i) * i  # ceil((m+1)/i) * i
            if first > m + L:
                return False  # No candidate for this i
            mask = 0
            for v in range(first, m + L + 1, i):
                pos = v - m - 1  # 0-indexed position
                mask |= (1 << pos)
            adj_masks[i - 1] = mask
        
        # Sort indices by number of candidates (most constrained first)
        order = sorted(range(n), key=lambda x: adj_masks[x].bit_count())
        
        # DFS with backtracking
        def dfs(idx, used):
            if idx == n:
                return True
            i = order[idx]
            avail = adj_masks[i] & ~used
            while avail:
                bit = avail & (-avail)
                if dfs(idx + 1, used | bit):
                    return True
                avail ^= bit
            return False
        
        return dfs(0, 0)
    
    def f_of(m):
        """Find minimum L such that can(m, L) is True."""
        if not can(m, MAXL):
            return -1  # Should not happen for valid m
        lo, hi = MINL, MAXL
        while lo < hi:
            mid = (lo + hi) // 2
            if can(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best_m = 0
    best_f = -1
    
    for m in range(PERIOD):
        if time.time() > deadline:
            break
        
        fv = f_of(m)
        if fv > best_f:
            best_f = fv
            best_m = m
            # If we found the theoretical maximum, we can stop early
            if best_f == MAXL:
                break
    
    return int(best_m)
