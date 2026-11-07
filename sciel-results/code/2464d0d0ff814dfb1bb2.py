import time
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match(m, L, n=12):
    # Check if (m, m+L] contains distinct b_1..b_n with i | b_i
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        v = r
        end = m + L
        while v <= end:
            adj[i - 1].append(v)
            v += i
        if not adj[i - 1]:
            return False
    
    # Order indices by number of candidates (most constrained first)
    # This helps pruning in the DFS
    indices = list(range(n))
    indices.sort(key=lambda x: len(adj[x]))
    
    # We will use a set to track used values
    used = set()
    
    def dfs(idx):
        if idx == n:
            return True
        
        # Get the i-th index in the sorted order
        i = indices[idx]
        
        for v in adj[i]:
            if v not in used:
                used.add(v)
                if dfs(idx + 1):
                    return True
                used.remove(v)
        
        return False
    
    return dfs(0)

def _f(m, n=12):
    # Binary search smallest L in [1, 144]
    lo, hi = 1, 144
    if not _can_match(m, hi, n):
        return hi # Should not happen per contract, but safe
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

    # The best m in [0, p-1] is equivalent to any m + k*p.
    # We return the smallest such m.
    return int(best_m)
