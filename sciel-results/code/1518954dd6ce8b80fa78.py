"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i|b_i."""
    lo = m + 1
    hi = m + L
    
    # For each i, find valid numbers
    valid = [None] * n
    for i in range(n):
        idx = i + 1
        first = ((lo + idx - 1) // idx) * idx
        if first > hi:
            return False
        valid[i] = []
        x = first
        while x <= hi:
            valid[i].append(x)
            x += idx
            
    # Sort indices by number of options (most constrained first)
    indices = list(range(n))
    indices.sort(key=lambda i: len(valid[i]))
    
    # Backtracking
    used = set()
    
    def backtrack(k):
        if k == n:
            return True
        orig_idx = indices[k]
        for x in valid[orig_idx]:
            if x not in used:
                used.add(x)
                if backtrack(k + 1):
                    return True
                used.remove(x)
        return False
    
    return backtrack(0)

def _compute_f(m, n=12, max_L=150):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search for the minimum L
    low = 1
    high = max_L
    ans = max_L
    
    # Quick check if max_L works? 
    # If not, return max_L (or handle error)
    # But f(12,m) is always <= 144.
    
    while low <= high:
        mid = (low + high) // 2
        if _has_matching(m, mid, n):
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
            
    return ans

def construct():
    """Return m in 0..1000000 that maximizes f(12, m)."""
    deadline = time.time() + 80  # Safety margin
    
    # Period is lcm(1..12) = 27720
    K = 27720
    best_m = 0
    best_L = 0
    
    for r in range(K):
        if time.time() > deadline:
            break
        L = _compute_f(r, n=12, max_L=150)
        if L > best_L:
            best_L = L
            best_m = r
            
    return int(best_m)
# EVOLVE-BLOCK-END
