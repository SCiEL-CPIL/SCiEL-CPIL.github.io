import time
import sys

# LCM of 1..12 = 27720
PERIOD = 27720
MAX_L = 200
N = 12

def _precompute_candidates(r):
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

def _can_match_offsets(cands, L):
    """Check if matching exists using offsets and integer bitmask DFS."""
    adj = []
    for i in range(N):
        valid = [off for off in cands[i] if off <= L]
        if not valid:
            return False
        adj.append(valid)
    
    match_to_idx = {}
    match_idx_to_off = [-1] * N

    def dfs(u, visited_mask):
        for off in adj[u]:
            bit = 1 << off
            if visited_mask & bit:
                continue
            new_mask = visited_mask | bit
            if off not in match_to_idx or dfs(match_to_idx[off], new_mask):
                match_to_idx[off] = u
                match_idx_to_off[u] = off
                return True
        return False

    for u in range(N):
        if not dfs(u, 0):
            return False
    return True

def _f_from_cands(cands):
    """Find smallest L using precomputed candidates."""
    if not _can_match_offsets(cands, MAX_L):
        return MAX_L
    lo, hi = 1, MAX_L
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match_offsets(cands, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = start + 80

    def elapsed():
        return time.time() - start

    best_f = 0
    best_residue = 0

    for r in range(PERIOD):
        if elapsed() > deadline:
            break
        cands = _precompute_candidates(r)
        f = _f_from_cands(cands)
        if f > best_f:
            best_f = f
            best_residue = r

    best_m = best_residue
    if best_m > 1000000:
        best_m = 0

    if elapsed() < deadline - 1:
        for dm in range(-5, 6):
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            cands = _precompute_candidates(mm % PERIOD)
            f = _f_from_cands(cands)
            if f > best_f:
                best_f = f
                best_m = mm

    return int(best_m)
