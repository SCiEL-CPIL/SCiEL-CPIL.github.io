import time

def construct():
    start_time = time.time()
    deadline = 89.0  # Safety margin before 90s limit
    
    n = 12
    max_m = 1000000
    
    # Precompute masks for each index i, each L, and each remainder r
    # precomputed_masks[i][L][r] is the bitmask for index i (1-based)
    # for window length L, with offset r (where p % i == r)
    
    precomputed_masks = [[[0] * (i+1) for i in range(n)] for L in range(145)]
    # Wait, structure: precomputed_masks[L][i][r]
    # L from 12 to 144
    # i from 0 to 11 (index 1 to 12)
    # r from 0 to i
    
    precomputed_masks = [None] * 145
    for L in range(12, 145):
        precomputed_masks[L] = [None] * n
        for i in range(n):
            idx = i + 1
            precomputed_masks[L][i] = [0] * idx
            for r in range(idx):
                mask = 0
                for p in range(r, L, idx):
                    mask |= (1 << p)
                precomputed_masks[L][i][r] = mask

    def can_assign(m, L):
        """
        Check if there's a perfect matching for indices 1..12 in (m, m+L].
        """
        # Get masks for this m and L
        # r_i = (-m - 1) % (i+1)
        
        avail_masks = [0] * n
        for i in range(n):
            idx = i + 1
            r = (-m - 1) % idx
            avail_masks[i] = precomputed_masks[L][i][r]
        
        # Check if any index has 0 options
        for mask in avail_masks:
            if mask == 0:
                return False

        # Recursive DFS
        def dfs(k, used_mask):
            if k == n:
                return True
            
            idx = n - 1 - k  # Check 12, 11, ..., 1
            mask = avail_masks[idx]
            
            m_mask = mask
            while m_mask:
                lsb = m_mask & (-m_mask)
                if not (used_mask & lsb):
                    if dfs(k + 1, used_mask | lsb):
                        return True
                m_mask ^= lsb
                
            return False
        
        return dfs(0, 0)

    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        lo, hi = 12, 144
        if not can_assign(m, hi):
            return 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    best_m = 0
    best_f = -1
    
    # Scan all m values
    for m in range(max_m + 1):
        if time.time() - start_time > deadline:
            break
        
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
