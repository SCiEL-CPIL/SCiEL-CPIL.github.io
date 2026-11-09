"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_range(n):
    return reduce(_lcm, range(1, n + 1))

LCM_PERIOD = _lcm_range(12)  # 27720

def _has_perfect_matching(adj, n):
    """Check if bipartite graph has perfect matching covering all n left nodes.
    adj[i] = list of right-side indices for left node i (0-indexed).
    Uses DFS-based augmenting path matching with array-based visited."""
    match_r = [-1] * 200  # right side match, indexed by position (0..199)
    
    def try_augment(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_r[v] == -1 or try_augment(match_r[v], visited):
                    match_r[v] = u
                    return True
        return False
    
    for u in range(n):
        visited = [False] * 200
        if not try_augment(u, visited):
            return False
    return True

def _compute_f_binary(m, n=12, max_L=144):
    """Compute f(n, m) using binary search on L."""
    # Quick check: does L=max_L work? If not, return max_L
    adj_max = []
    for i in range(1, n + 1):
        candidates = []
        for j in range(max_L):
            if (m + 1 + j) % i == 0:
                candidates.append(j)
        adj_max.append(candidates)
        if not candidates:
            return max_L  # Can't even cover for max_L
    
    if not _has_perfect_matching(adj_max, n):
        return max_L
    
    # Binary search for minimum L
    lo, hi = 1, max_L
    while lo < hi:
        mid = (lo + hi) // 2
        adj = []
        for i in range(1, n + 1):
            candidates = []
            for j in range(mid):
                if (m + 1 + j) % i == 0:
                    candidates.append(j)
            adj.append(candidates)
            if not candidates:
                break
        if not adj or len(adj) < n:
            lo = mid + 1
        else:
            if _has_perfect_matching(adj, n):
                hi = mid
            else:
                lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Precompute f(12, r) for all residues r in [0, LCM_PERIOD)
    best_f = 0
    best_residue = 0
    
    for r in range(LCM_PERIOD):
        if time.time() > deadline:
            break
        f_val = _compute_f_binary(r, n=12, max_L=144)
        if f_val > best_f:
            best_f = f_val
            best_residue = r
    
    # Find m in [0, 1000000] with m % LCM_PERIOD == best_residue
    if best_residue <= 1000000:
        m = best_residue
    else:
        m = best_residue
    
    # Ensure m is a Python int
    m = int(m)
    
    # Verify m is in range
    if m < 0 or m > 1000000:
        m = 0
    
    return m

# EVOLVE-BLOCK-END
