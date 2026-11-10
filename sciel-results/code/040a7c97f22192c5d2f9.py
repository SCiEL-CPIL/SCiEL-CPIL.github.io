import time
from functools import lru_cache
from math import gcd

# EVOLVE-BLOCK-START
P = {'n': 12, 'maximum_start': 1000000}
_LCM = 27720  # lcm(1..12)

def _has_perfect_matching(m, L, n=12):
    """Check if (m, m+L] contains n distinct b_i with i|b_i."""
    if L < n:
        return False
    # Build adjacency: for each i in 1..n, which numbers in (m, m+L] are divisible by i
    adj = [[] for _ in range(n)]
    for j in range(L):
        num = m + 1 + j
        for i in range(1, n+1):
            if num % i == 0:
                adj[i-1].append(j)
    # Bipartite matching via DFS (Kuhn's algorithm)
    match_r = [-1] * L
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    result = 0
    for u in range(n):
        visited = [False] * L
        if dfs(u, visited):
            result += 1
    return result == n

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has n distinct b_i with i|b_i."""
    lo, hi = 1, max_L
    if not _has_perfect_matching(m, max_L, n):
        return max_L + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 85
    n = P['n']
    max_start = P['maximum_start']
    
    best_f = 0
    best_m = 0
    
    # Phase 1: Search over residues mod LCM
    for r in range(_LCM):
        if time.time() > deadline:
            break
        f_val = _compute_f(r, n)
        if f_val > best_f:
            best_f = f_val
            best_m = r
    
    # Phase 2: Local refinement around the best residue
    # Check m values near best_m that might have higher f
    # Since f depends on m mod LCM, we check nearby values
    # Also check some strategic m values
    
    # Check m values in a window around best_m
    search_radius = 50
    for dm in range(-search_radius, search_radius + 1):
        if time.time() > deadline:
            break
        m_candidate = best_m + dm
        if 0 <= m_candidate <= max_start:
            f_val = _compute_f(m_candidate, n)
            if f_val > best_f:
                best_f = f_val
                best_m = m_candidate
    
    # Phase 3: Check some strategic m values
    # m values that are multiples of small numbers might create harder constraints
    strategic_ms = []
    for k in range(1, 100):
        strategic_ms.append(k * _LCM)
    for k in range(1, 50):
        strategic_ms.append(k * 720)  # lcm(1..10)
    for k in range(1, 50):
        strategic_ms.append(k * 840)  # lcm(1..12) / some factor
    
    for m_candidate in strategic_ms:
        if time.time() > deadline:
            break
        if 0 <= m_candidate <= max_start:
            f_val = _compute_f(m_candidate, n)
            if f_val > best_f:
                best_f = f_val
                best_m = m_candidate
    
    return int(best_m)
# EVOLVE-BLOCK-END
