import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    N = 12
    LCM_PERIOD = 27720
    MAX_L = 144
    
    def can_match(m, L):
        # Precompute valid x for each i
        # x in [0, L-1] corresponds to b = m + 1 + x
        # i | b  <=>  i | (m + 1 + x)  <=>  x = -1 - m (mod i)
        r = [(-1 - m) % i for i in range(1, N+1)]
        
        # valid_x[i] is a list of x in [0, L-1] such that x % i == r[i]
        # We want to order by length of valid_x
        valid_x = []
        for i in range(1, N+1):
            ri = r[i-1]
            # x = ri, ri+i, ri+2i, ... < L
            # Note: ri is in [0, i-1]
            if ri >= L:
                # No solution for this i
                return False
            # Generate list
            v = []
            x = ri
            while x < L:
                v.append(x)
                x += i
            valid_x.append(v)
            
        # Order indices by number of candidates
        order = sorted(range(N), key=lambda i: len(valid_x[i]))
        
        # Backtracking with bitmask for used x
        # L is max 144, so bitmask fits in int
        used_mask = 0
        
        def backtrack(idx):
            nonlocal used_mask
            if idx == N:
                return True
            i = order[idx]
            for x in valid_x[i]:
                bit = 1 << x
                if not (used_mask & bit):
                    used_mask |= bit
                    if backtrack(idx + 1):
                        return True
                    used_mask &= ~bit
            return False
        
        return backtrack(0)

    def compute_f(m):
        lo, hi = 1, MAX_L
        # Check if hi is enough
        if not can_match(m, hi):
            # Try larger, though 144 should be enough
            while hi <= 200 and not can_match(m, hi):
                hi += 1
            if not can_match(m, hi):
                return 0
        
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    best_m = 0
    best_f = 0
    
    # Search over one period
    for m in range(LCM_PERIOD):
        if time.time() > deadline:
            break
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
