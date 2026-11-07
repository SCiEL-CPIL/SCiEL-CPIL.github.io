import time
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _bipartite_match(adj, n_left, n_right):
    """Kuhn's algorithm for bipartite matching. adj[i] = list of right-node indices."""
    match_r = [-1] * n_right  # match_r[j] = left node matched to right j, or -1
    
    def try_kuhn(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or try_kuhn(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(n_left):
        visited = [False] * n_right
        try_kuhn(u, visited)
    
    return sum(1 for v in match_r if v != -1)

def _can_match(m, L, n=12):
    if L < n:
        return False
        
    val_to_idx = {}
    right_vals = []
    adj_idx = [[] for _ in range(n)]
    
    for i in range(n):
        idx = i + 1
        start_k = m // idx + 1
        end_k = (m + L) // idx
        
        if start_k > end_k:
            return False
            
        for k in range(start_k, end_k + 1):
            v = k * idx
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
            adj_idx[i].append(val_to_idx[v])
            
    if len(right_vals) < n:
        return False
    
    # Quick check: if any left node has no edges, fail
    for i in range(n):
        if not adj_idx[i]:
            return False
    
    return _bipartite_match(adj_idx, n, len(right_vals)) == n

def _f(m, n=12):
    # Binary search smallest L in [12, 144]
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Calculate LCM of 1..12
    p = reduce(_lcm, range(1, 13))
    
    # First pass: check critical m values (m+1 divisible by some i in 2..12)
    # These are most likely to have high f values
    best_m = 0
    best_f = _f(0)
    
    # Generate candidates: m where m+1 is divisible by at least one of 2..12
    # i.e., m ≡ -1 (mod i) for some i in 2..12
    # This means m+1 is NOT coprime to lcm(2..12)
    # Actually, let's just check all m but prioritize those where m+1 shares factors with 2..12
    
    # Strategy: check m values in order of "difficulty"
    # m where m+1 is divisible by many small numbers first
    # Since f(12,m) is periodic with period p=27720, check one full period
    
    # Precompute for each m in [0, p-1] how many i in 2..12 divide m+1
    # Higher count = more likely to be "hard" = higher f
    scores = []
    for m in range(p):
        mp1 = m + 1
        cnt = 0
        for i in range(2, 13):
            if mp1 % i == 0:
                cnt += 1
        scores.append((cnt, m))
    
    # Sort by descending difficulty
    scores.sort(key=lambda x: -x[0])
    
    for cnt, m in scores:
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        # Early termination: if we've found the maximum possible value
        if best_f >= 144:
            break
    
    return int(best_m)
