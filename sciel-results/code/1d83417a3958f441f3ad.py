import time
import sys

# LCM of 1..12 = 27720
PERIOD = 27720

def _can_match_fast(m, L, n=12):
    """Check if (m, m+L] has distinct b_1..b_n with i | b_i using bitmask matching."""
    end = m + L
    # For each index i, compute the set of candidate values as bitmasks
    # Values are in range (m, m+L], so at most L values
    # We'll use a list of sets for simplicity first, then optimize
    adj = [set() for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        if r > end:
            return False
        v = r
        while v <= end:
            adj[i - 1].add(v)
            v += i
    
    # Bitmask-based bipartite matching using DFS with visited as a set
    # Since n=12 is small, we can use a simple approach
    match_to_val = [-1] * n  # match_to_val[i] = value matched to index i, or -1
    match_val_to_idx = {}    # value -> index
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_val_to_idx or dfs(match_val_to_idx[v], visited):
                match_val_to_idx[v] = u
                match_to_val[u] = v
                return True
        return False
    
    for u in range(n):
        visited = set()
        if not dfs(u, visited):
            return False
    return True

def _f(m, n=12):
    """Find smallest L such that (m, m+L] has valid matching."""
    if not _can_match_fast(m, 200, n):
        return 200
    lo, hi = 1, 200
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_fast(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin
    
    def elapsed():
        return time.time() - start
    
    # Key insight: f(12, m) depends only on m mod 27720
    # So we can search all residues 0..27719 exhaustively
    # This guarantees finding the global maximum
    
    best_f = 0
    best_residue = 0
    
    # Exhaustive search over one period
    for r in range(PERIOD):
        if elapsed() > deadline:
            break
        f = _f(r)
        if f > best_f:
            best_f = f
            best_residue = r
    
    # Find the smallest m in [0, 1000000] with m % 27720 == best_residue
    # m = best_residue, or best_residue + 27720*k for some k >= 0
    best_m = best_residue
    if best_m > 1000000:
        # Shouldn't happen since best_residue < 27720
        best_m = 0
    
    # But we should also check if there's a better m nearby that wasn't in [0, PERIOD)
    # Actually, since f is periodic, all m with the same residue have the same f
    # So best_m = best_residue is optimal (and it's in [0, 1000000])
    
    # Fine-tune: check a few values around best_m to be safe
    if elapsed() < deadline - 1:
        for dm in range(-5, 6):
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    return int(best_m)
