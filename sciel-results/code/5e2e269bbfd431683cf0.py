"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}
LCM_12 = 27720  # lcm(1,2,...,12)

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i using Hopcroft-Karp."""
    if L < n:
        return False
    
    # Build candidate offsets (0-indexed) for each index i
    # Offset o (1..L) is valid for i if (m+o) % i == 0
    # o % i == (-m) % i
    adj = []
    for i in range(1, n + 1):
        target_res = (-m) % i
        start = i if target_res == 0 else target_res
        cands = []
        for o in range(start, L + 1, i):
            cands.append(o - 1)  # 0-indexed
        if not cands:
            return False
        adj.append(cands)
    
    # Hopcroft-Karp algorithm
    pair_u = [-1] * n  # which offset is matched to each left node
    pair_v = [-1] * L  # which left node is matched to each right node
    dist = [0] * n
    
    def bfs():
        queue = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = -1
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    queue.append(pu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False
    
    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    return matching == n

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] contains distinct b_i with i|b_i."""
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
    deadline = time.time() + 85
    best_m = 12
    best_f = _compute_f(best_m)
    
    # Phase 1: Coarse scan - check every 4th residue to find promising regions
    step = 4
    for m in range(0, LCM_12, step):
        if time.time() > deadline - 15:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f >= 144:
                return int(best_m)
    
    # Phase 2: Fine scan around best found so far
    search_radius = 30
    lo_bound = max(0, best_m - search_radius)
    hi_bound = min(LCM_12, best_m + search_radius + 1)
    for m in range(lo_bound, hi_bound):
        if time.time() > deadline - 10:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f >= 144:
                return int(best_m)
    
    # Phase 3: Full scan if time permits
    if time.time() < deadline - 20:
        for m in range(0, LCM_12):
            if time.time() > deadline - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                if best_f >= 144:
                    return int(best_m)
    
    # Phase 4: Local refinement around best
    if time.time() < deadline - 10:
        radius = 50
        lo_bound = max(0, best_m - radius)
        hi_bound = min(LCM_12, best_m + radius + 1)
        for m in range(lo_bound, hi_bound):
            if time.time() > deadline - 5:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
                if best_f >= 144:
                    return int(best_m)
    
    return int(best_m)
# EVOLVE-BLOCK-END
