import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    N = 12
    PERIOD = 27720  # lcm(1,2,...,12)
    MAX_L = 144
    
    # Precompute candidate bitmasks for each (i, r) where r = m % i
    # mask[i][r] has bit (d-1) set if d is a valid offset for i given m ≡ r (mod i)
    # We only need d up to MAX_L
    cand_mask_table = [[0] * i for i in range(N + 1)]
    for i in range(1, N + 1):
        for r in range(i):
            d = (-r) % i
            if d == 0:
                d = i
            mask = 0
            while d <= MAX_L:
                mask |= (1 << (d - 1))
                d += i
            cand_mask_table[i][r] = mask
    
    def has_matching(m, L):
        # Build candidate masks limited to L
        cands = [0] * (N + 1)
        cands[0] = 0
        for i in range(1, N + 1):
            r = m % i
            full_mask = cand_mask_table[i][r]
            # Limit to positions 1..L
            limit_mask = (1 << L) - 1
            cands[i] = full_mask & limit_mask
            if cands[i] == 0:
                return False
        
        # Sort by popcount for most-constrained-first
        order = list(range(1, N + 1))
        order.sort(key=lambda i: cands[i].bit_count())
        
        # Backtracking with bitsets and forward checking
        used = 0
        
        def bt(idx):
            nonlocal used
            if idx == N:
                return True
            i = order[idx]
            avail = cands[i] & ~used
            while avail:
                lowbit = avail & (-avail)
                avail ^= lowbit
                used |= lowbit
                # Forward check
                feasible = True
                for j in range(idx + 1, N):
                    if (cands[order[j]] & ~used) == 0:
                        feasible = False
                        break
                if feasible and bt(idx + 1):
                    return True
                used ^= lowbit
            return False
        
        return bt(0)
    
    best_m = 0
    best_L = 0
    
    for r in range(PERIOD):
        if time.time() - start_time > deadline:
            break
        
        lo, hi = 1, MAX_L
        while lo < hi:
            mid = (lo + hi) // 2
            if has_matching(r, mid):
                hi = mid
            else:
                lo = mid + 1
        
        if lo > best_L:
            best_L = lo
            best_m = r
            if best_L == 144:
                break
    
    return int(best_m)
