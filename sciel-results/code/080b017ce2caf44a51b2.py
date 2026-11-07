"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _compute_f(m, L, n=12):
    """Check if interval (m, m+L] has distinct b_1..b_n with i|b_i."""
    # For each i, collect multiples of i in (m, m+L]
    # A multiple of i in (m, m+L] is m + k*i where k>=1 and m+k*i <= m+L, i.e., k*i <= L
    # So multiples are m+i, m+2i, ..., m+floor(L/i)*i
    # We need to assign distinct values to each i.
    # This is a bipartite matching: left side = {1..n}, right side = set of all candidate values.
    # But the right side can have at most sum(floor(L/i)) values, and we need n distinct ones.
    
    # Build adjacency: for each i, the set of multiples in (m, m+L]
    # Use Hopcroft-Karp or simple DFS since n=12 is tiny
    
    # Collect all candidate values
    adj = {}  # i -> list of values
    for i in range(1, n+1):
        vals = []
        k = 1
        while k * i <= L:
            vals.append(m + k * i)
            k += 1
        adj[i] = vals
    
    # If any i has no multiples, return False
    for i in range(1, n+1):
        if not adj[i]:
            return False
    
    # Simple bipartite matching (DFS-based, sufficient for n=12)
    # Right side: all distinct values
    right_vals = set()
    for i in range(1, n+1):
        for v in adj[i]:
            right_vals.add(v)
    
    # Map right values to indices
    val_to_idx = {v: idx for idx, v in enumerate(sorted(right_vals))}
    num_right = len(right_vals)
    
    # match_right[j] = left node matched to right node j
    match_right = [-1] * num_right
    
    def dfs(u, visited):
        for v in adj[u]:
            j = val_to_idx[v]
            if visited[j]:
                continue
            visited[j] = True
            if match_right[j] == -1 or dfs(match_right[j], visited):
                match_right[j] = u
                return True
        return False
    
    count = 0
    for u in range(1, n+1):
        visited = [False] * num_right
        if dfs(u, visited):
            count += 1
        else:
            return False
    return count == n

def _compute_f_min_L(m, n=12, max_L=144):
    """Find the least L such that (m, m+L] has the matching."""
    for L in range(1, max_L + 1):
        if _compute_f(m, L, n):
            return L
    return max_L + 1  # should not happen per problem constraints

def construct():
    start_time = time.time()
    deadline = 75.0  # safety margin before 90s
    
    n = 12
    # Period is lcm(1,2,...,12) = 27720
    period = 1
    for i in range(1, n+1):
        period = _lcm(period, i)
    
    max_start = 1000000
    max_L = 144
    
    best_m = 0
    best_L = 0
    
    # Check all residues 0..period-1
    for r in range(period):
        if time.time() > deadline:
            break
        L_val = _compute_f_min_L(r, n, max_L)
        if L_val > best_L:
            best_L = L_val
            best_m = r
    
    # Find the largest m <= max_start with m % period == best_m
    if best_m <= max_start:
        # m = best_m + k*period <= max_start
        # k <= (max_start - best_m) / period
        k = (max_start - best_m) // period
        result = best_m + k * period
    else:
        result = best_m  # shouldn't happen since best_m < period <= 27720 < 1000000
    
    return int(result)
# EVOLVE-BLOCK-END
