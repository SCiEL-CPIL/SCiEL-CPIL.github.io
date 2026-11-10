import time
from math import gcd

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_range(n):
    result = 1
    for i in range(1, n + 1):
        result = _lcm(result, i)
    return result

def _can_match(m, L, n=12):
    """Check if there exist distinct b_1..b_n in (m, m+L] with i|b_i."""
    # Build candidate lists for each i
    candidates = []
    for i in range(1, n + 1):
        # multiples of i in (m, m+L]
        # first multiple > m: m//i * i, if that's <= m, add i
        first = (m // i + 1) * i
        if first > m + L:
            return False
        cands = []
        v = first
        while v <= m + L:
            cands.append(v)
            v += i
        candidates.append(cands)
    
    # Bipartite matching: left = i (0..n-1), right = values
    # Use DFS augmenting path
    # Map values to indices
    all_vals = set()
    for cands in candidates:
        all_vals.update(cands)
    val_to_idx = {v: idx for idx, v in enumerate(sorted(all_vals))}
    num_right = len(all_vals)
    
    # Adjacency: for each left i, list of right indices
    adj = []
    for i in range(n):
        adj.append([val_to_idx[v] for v in candidates[i]])
    
    matchR = [-1] * num_right
    matchL = [-1] * n
    
    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if matchR[v] == -1 or dfs(matchR[v], seen):
                matchR[v] = u
                matchL[u] = v
                return True
        return False
    
    for u in range(n):
        seen = [False] * num_right
        if not dfs(u, seen):
            return False
    return True

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L
    lo, hi = 1, max_L
    best = max_L + 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin
    
    n = 12
    period = _lcm_range(n)  # 27720
    
    best_m = 0
    best_f = 0
    
    for m in range(period):
        if time.time() > deadline:
            break
        f_val = _compute_f(m, n)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return best_m
