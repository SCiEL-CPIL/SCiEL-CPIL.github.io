import time
import math
from functools import reduce

def _kuhn_match(adj, n_left, n_right):
    """Kuhn's algorithm: DFS augmenting paths. Fast for small graphs."""
    pair_v = [-1] * n_right
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if pair_v[v] == -1 or dfs(pair_v[v], visited):
                pair_v[v] = u
                return True
        return False
    
    matching = 0
    for u in range(n_left):
        visited = [False] * n_right
        if dfs(u, visited):
            matching += 1
    return matching

def _can_match(m, L, n=12):
    if L < n:
        return False
    
    adj = [[] for _ in range(n)]
    val_to_idx = {}
    right_count = 0
    
    for i in range(n):
        idx = i + 1
        start_k = m // idx + 1
        end_k = (m + L) // idx
        
        if start_k > end_k:
            return False
        
        for k in range(start_k, end_k + 1):
            v = k * idx
            if v not in val_to_idx:
                val_to_idx[v] = right_count
                right_count += 1
            adj[i].append(val_to_idx[v])
    
    if right_count < n:
        return False
    
    # Quick Hall check: each left node needs at least one neighbor
    for i in range(n):
        if not adj[i]:
            return False
    
    return _kuhn_match(adj, n, right_count) == n

def _f(m, n=12):
    # Binary search smallest L in [12, 144]
    lo, hi = 12, 143
    if not _can_match(m, hi, n):
        return 144
        
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
    
    # LCM of 1..12 = 27720
    p = 27720
    
    # Check all m in [0, p-1] since f(12, m) is periodic with period p
    best_m = 0
    best_f = _f(0)
    
    for m in range(1, p):
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
