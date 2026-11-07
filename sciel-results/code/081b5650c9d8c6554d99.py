"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 12, 'maximum_start': 1000000}

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i."""
    if L < n:
        return False
    
    # Build candidate offsets (0-indexed) for each index i
    # Offset o (1..L) is valid for i if (m+o) % i == 0
    # o % i == (-m) % i
    candidates = []
    for i in range(1, n + 1):
        target_res = (-m) % i
        if target_res == 0:
            start = i
        else:
            start = target_res
            
        cands = []
        for o in range(start, L + 1, i):
            cands.append(o - 1) # 0-indexed for bitmask
            
        if not cands:
            return False
        candidates.append(cands)
        
    # Order indices by scarcity (fewest candidates first)
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    
    # Backtracking DFS with bitmask for used offsets, with memoization
    memo = set()
    
    def dfs(idx, used_mask):
        if idx == n:
            return True
        state = (idx, used_mask)
        if state in memo:
            return False
        i = order[idx]
        for o_idx in candidates[i]:
            bit = 1 << o_idx
            if not (used_mask & bit):
                if dfs(idx + 1, used_mask | bit):
                    return True
        memo.add(state)
        return False
        
    return dfs(0, 0)

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] contains distinct b_i with i|b_i."""
    # Binary search on L since _has_matching is monotone in L
    lo, hi = n, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 85  # 85s deadline with 5s safety margin
    best_m = 12
    best_f = _compute_f(best_m)
    
    # The key insight: f(12, m) depends on the pattern of multiples of 1..12 in (m, m+L].
    # This pattern is periodic with period lcm(1,2,...,12) = 27720.
    # So we only need to search m mod 27720, i.e., m in [0, 27719].
    
    lcm_12 = 27720
    
    # Search all residues mod 27720
    for m in range(0, lcm_12):
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f >= 144:
                break
    
    # If we found a good residue, verify it works for larger m values too
    # (in case there's some non-periodic behavior, though there shouldn't be)
    if time.time() < deadline - 10 and best_f < 144:
        best_residue = best_m % lcm_12
        # Check a few larger m values with the same residue
        for k in range(1, 40):
            m = k * lcm_12 + best_residue
            if m > 1000000:
                break
            if time.time() > deadline - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                if best_f >= 144:
                    break
    
    # Local search around the best m found
    if time.time() < deadline - 10:
        search_range = max(0, best_m - 200)
        search_end = min(1000001, best_m + 200)
        for m in range(search_range, search_end):
            if time.time() > deadline - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                if best_f >= 144:
                    break
    
    return best_m
# EVOLVE-BLOCK-END
