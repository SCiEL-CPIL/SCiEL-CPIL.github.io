"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 12, 'maximum_start': 1000000}

def _has_perfect_matching(m, L):
    """Check if in (m, m+L] we can assign distinct b_i to each i in 1..12 with i|b_i."""
    n = 12
    
    # Build adjacency list for left side (indices 1..12)
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        first = m + 1
        rem = first % i
        if rem == 0:
            first_mult = first
        else:
            first_mult = first + (i - rem)
        for b in range(first_mult, m + L + 1, i):
            idx = b - m - 1
            if 0 <= idx < L:
                adj[i - 1].append(idx)
    
    # DFS-based bipartite matching
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
    
    for u in range(n):
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    lcm_12 = 27720
    max_m = 1000000
    
    # Key insight: f(12, m) depends only on m mod lcm_12
    # So we only need to evaluate m in [0, lcm_12 - 1]
    # Then find the largest m <= max_m with the best residue
    
    best_f = 0
    best_residue = 0
    
    for r in range(lcm_12):
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_residue = r
    
    # Find the largest m <= max_m such that m % lcm_12 == best_residue
    # m = best_residue + k * lcm_12, maximize k such that m <= max_m
    k_max = (max_m - best_residue) // lcm_12
    if k_max < 0:
        # best_residue > max_m, shouldn't happen since lcm_12 > max_m? No, 27720 < 1000000
        # Actually best_residue < lcm_12 = 27720 < 1000000, so k_max >= 0
        best_m = best_residue
    else:
        best_m = best_residue + k_max * lcm_12
    
    # Verify best_m is within bounds
    if best_m > max_m:
        best_m = max_m
    
    return int(best_m)
# EVOLVE-BLOCK-END
