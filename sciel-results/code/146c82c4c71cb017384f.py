import time

def construct():
    start_time = time.time()
    limit = 75  # safety margin
    
    N = 12
    period = 27720  # lcm(1..12)
    
    best_m = 0
    best_f = 0
    
    def get_candidates(m, L, i):
        """Return list of x in (m, m+L] divisible by i"""
        start = m + 1
        rem = start % i
        if rem == 0:
            first = start
        else:
            first = start + (i - rem)
        
        res = []
        for x in range(first, m + L + 1, i):
            res.append(x)
        return res
    
    def can_match(m, L):
        # Build candidate lists for i=1..12
        cands = []
        for i in range(1, N + 1):
            cands.append(get_candidates(m, L, i))
        
        # Check if any i has no candidates
        for i in range(N):
            if len(cands[i]) == 0:
                return False
        
        # Build adjacency: for each i, which values it can match
        # Use bitmask for efficiency
        # First, collect all unique values
        all_vals = set()
        for i in range(N):
            all_vals.update(cands[i])
        all_vals = sorted(all_vals)
        val_to_idx = {v: idx for idx, v in enumerate(all_vals)}
        
        # Build bitmask for each i
        masks = [0] * N
        for i in range(N):
            mask = 0
            for x in cands[i]:
                mask |= (1 << val_to_idx[x])
            masks[i] = mask
        
        # Now use backtracking with bitmask
        indices = list(range(N))
        indices.sort(key=lambda idx: bin(masks[idx]).count('1'))
        
        def backtrack(k, used_mask):
            if k == N:
                return True
            i = indices[k]
            available = masks[i] & ~used_mask
            while available:
                bit = available & (-available)
                available -= bit
                if backtrack(k + 1, used_mask | bit):
                    return True
            return False
        
        return backtrack(0, 0)
    
    def compute_f(m):
        # Binary search on L
        lo = 1
        hi = 200
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Search m in one period
    check_interval = 100
    
    for m in range(period):
        if m % check_interval == 0:
            if time.time() - start_time > limit:
                break
        
        f_val = compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
