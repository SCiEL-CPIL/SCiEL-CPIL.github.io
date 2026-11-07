"""Find worst interval start for distinct-multiple matching with n=12."""
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_lcm_table(n):
    L = 1
    for i in range(2, n + 1):
        L = _lcm(L, i)
    return L

def _can_match_fast(m, L, n):
    """
    Check if (m, m+L] contains distinct b_i with i | b_i for i=1..n.
    Uses bitmask backtracking for speed on small N.
    """
    # The numbers in the interval are m+1, ..., m+L
    # We map them to indices 0 to L-1
    # idx = val - (m+1)
    
    # Precompute masks for each i (1 to n)
    # mask[i] has bit j set if (m+1+j) is divisible by i
    masks = [0] * (n + 1)
    
    for i in range(1, n + 1):
        mask = 0
        # First multiple of i in (m, m+L]
        first = (m // i + 1) * i
        # Check each multiple
        k = first
        while k <= m + L:
            # Index of k in the interval
            idx = k - (m + 1)
            if 0 <= idx < L:
                mask |= (1 << idx)
            k += i
        masks[i] = mask
        if mask == 0:
            return False

    # Backtracking to assign indices to i=1..n
    # used_mask: bitmask of indices already assigned
    # i: current index to assign (1-based)
    
    # Order of assignment can affect speed. 
    # i with fewer options (popcount(mask[i])) should be tried first?
    # For n=12, simple order 1..12 is usually fine, but let's sort by popcount to be safe.
    # However, the constraint is "i divides b_i". The index i is fixed.
    # We can't change which i gets which number, but we can change the *order* in which we try to assign them in the DFS.
    
    # Let's stick to 1..n for simplicity, as n=12 is small.
    
    def dfs(i, used_mask):
        if i > n:
            return True
        # Get available bits for i
        avail = masks[i] & ~used_mask
        while avail:
            # Pick the lowest set bit
            lsb = avail & -avail
            avail ^= lsb
            if dfs(i + 1, used_mask | lsb):
                return True
        return False
    
    return dfs(1, 0)

def _compute_f(m, n, max_L=100):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search for the smallest L
    lo = 1
    hi = max_L
    ans = max_L
    
    # Quick check if even max_L works
    if not _can_match_fast(m, max_L, n):
        return max_L # Or handle failure, but task implies solution exists or max_L is bound
        
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(m, mid, n):
            hi = mid
            ans = mid
        else:
            lo = mid + 1
            
    return ans

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    n = 12
    LCM = _build_lcm_table(n)  # 27720
    
    best_m = 0
    best_f = 0
    
    # Precompute f(m) for all residue classes mod LCM
    for r in range(LCM):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
            
    return int(best_m)

construct()
