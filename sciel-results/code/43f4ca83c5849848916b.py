"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

PERIOD = 27720  # lcm(1,2,...,12)
N = 12
MAX_L = 144

def _precompute_offsets(r):
    """Precompute candidate offsets for each index i in (r, r+MAX_L]."""
    cands = []
    for i in range(1, N + 1):
        first = ((r // i) + 1) * i
        vals = []
        v = first
        while v <= r + MAX_L:
            vals.append(v - r)
            v += i
        cands.append(vals)
    return cands

def _can_match(cands, L):
    """Check if perfect matching exists using precomputed offsets."""
    # Filter candidates to those <= L
    adj = []
    for i in range(N):
        valid = [off for off in cands[i] if off <= L]
        if not valid:
            return False
        adj.append(valid)
    
    # DFS bipartite matching
    match_to_idx = {}
    
    def dfs(u, visited):
        for off in adj[u]:
            if off in visited:
                continue
            visited.add(off)
            if off not in match_to_idx or dfs(match_to_idx[off], visited):
                match_to_idx[off] = u
                return True
        return False
    
    for u in range(N):
        visited = set()
        if not dfs(u, visited):
            return False
    return True

def _f_from_cands(cands):
    """Find smallest L using precomputed candidates with binary search."""
    if not _can_match(cands, MAX_L):
        return MAX_L
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(cands, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 80  # Safety margin before 90s kill
    
    best_L = 0
    best_r = 0
    
    for r in range(PERIOD):
        if time.time() > deadline - 5:
            break
        cands = _precompute_offsets(r)
        f = _f_from_cands(cands)
        if f > best_L:
            best_L = f
            best_r = r
    
    # Find m in [0, 1000000] with m % PERIOD == best_r
    m = best_r
    if m > 1000000:
        m = 0
    
    return int(m)
# EVOLVE-BLOCK-END
