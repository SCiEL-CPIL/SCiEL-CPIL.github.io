import time
import sys

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses bitset-based DFS backtracking for speed.
    """
    # Precompute bitmasks for each i in 1..n
    # mask[i] has bit j set if (m + 1 + j) is divisible by i
    masks = [0] * (n + 1)
    for i in range(1, n + 1):
        mask = 0
        for j in range(L):
            val = m + 1 + j
            if val % i == 0:
                mask |= (1 << j)
        masks[i] = mask
        
    # If any i has no options, return False
    for i in range(1, n + 1):
        if masks[i] == 0:
            return False
            
    # Sort i by number of options (most constrained first) to prune early
    indices = list(range(1, n + 1))
    indices.sort(key=lambda i: bin(masks[i]).count('1'))
    
    def _dfs(k, current_used_mask):
        if k == n:
            return True
        i = indices[k]
        avail = masks[i] & ~current_used_mask
        while avail:
            bit = avail & -avail
            avail -= bit
            if _dfs(k + 1, current_used_mask | bit):
                return True
        return False
        
    return _dfs(0, 0)

def _f_val(m, n=12):
    """
    Find the least L such that (m, m+L] allows a matching.
    """
    # Max L is known to be small, 200 is a safe upper bound
    for L in range(1, 201):
        if _can_match(m, L, n):
            return L
    return 200 # Fallback, shouldn't happen

def construct():
    start_time = time.time()
    
    # Period is LCM(1..12) = 27720
    period = 27720
    max_m = 1000000
    
    best_L = 0
    best_residue = 0
    
    # We only need to check one period
    for r in range(period):
        if time.time() - start_time > 80:
            break
            
        L = _f_val(r)
        if L > best_L:
            best_L = L
            best_residue = r
            
    # Find the largest m <= max_m such that m % period == best_residue
    # m = k * period + best_residue
    # k * period <= max_m - best_residue
    # k <= (max_m - best_residue) // period
    if max_m >= best_residue:
        k = (max_m - best_residue) // period
        best_m = k * period + best_residue
    else:
        best_m = best_residue
        
    # Ensure best_m is within bounds
    if best_m > max_m:
        best_m = max_m
        
    return int(best_m)
