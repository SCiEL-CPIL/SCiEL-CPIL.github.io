import time

def _can_match_bitset(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1..b_n with i | b_i.
    Uses bitmask backtracking for speed.
    """
    # masks[i] is a bitmask where bit k is 1 if k is a valid candidate for index i
    # k ranges from 1 to L
    masks = [0] * n
    
    for i in range(1, n + 1):
        r = (-m) % i
        # Find first k >= 1 such that k % i == r
        if r == 0:
            # k must be multiple of i. Smallest positive is i.
            first_k = i
        else:
            first_k = r
            
        mask = 0
        # Set bits for all valid k
        # k, k+i, k+2i... <= L
        for k in range(first_k, L + 1, i):
            mask |= (1 << k)
        masks[i - 1] = mask
        
    # Check if any mask is 0
    for mask in masks:
        if mask == 0:
            return False
            
    # Sort indices by number of candidates (popcount) to prune early
    # We need to keep track of original index to get the mask
    indexed_masks = [(masks[i], i) for i in range(n)]
    indexed_masks.sort(key=lambda x: bin(x[0]).count('1'))
    
    # Backtracking
    # used_mask tracks which k values have been assigned
    # idx tracks how many indices we've assigned
    # We process indices in sorted order
    
    def dfs(idx, used_mask):
        if idx == n:
            return True
        
        mask, orig_idx = indexed_masks[idx]
        # Available candidates are those in mask not in used_mask
        available = mask & ~used_mask
        
        while available:
            # Get the lowest set bit
            lsb = available & -available
            # Clear the bit
            available ^= lsb
            # Recurse
            if dfs(idx + 1, used_mask | lsb):
                return True
        return False
        
    return dfs(0, 0)

def _f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_bitset(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 12
    period = 27720  # lcm(1, 2, ..., 12)
    
    best_m = 0
    best_f = 0
    
    # We can optimize by checking m values.
    # Since f(m) is periodic with period 27720, we just check 0..27719.
    for m in range(period):
        if time.time() > deadline:
            break
            
        # Quick check: if best_f is already high, we might skip some?
        # No, we need to find the max.
        
        f_val = _f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
