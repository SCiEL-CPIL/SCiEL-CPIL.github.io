import math
import time
from collections import deque
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _hopcroft_karp(adj, n_left, n_right):
    """
    Hopcroft-Karp algorithm for bipartite matching.
    adj: list of lists, adj[i] contains indices of right nodes connected to left node i.
    n_left: number of left nodes.
    n_right: number of right nodes.
    Returns the size of the maximum matching.
    """
    pair_u = [-1] * n_left  # pair_u[i] = right node matched to left node i
    pair_v = [-1] * n_right # pair_v[j] = left node matched to right node j
    dist = [0] * n_left
    
    def bfs():
        queue = deque()
        for u in range(n_left):
            if pair_u[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = float('inf')
        
        found = False
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if pair_v[v] == -1:
                    found = True
                elif dist[pair_v[v]] == float('inf'):
                    dist[pair_v[v]] = dist[u] + 1
                    queue.append(pair_v[v])
        return found

    def dfs(u):
        for v in adj[u]:
            if pair_v[v] == -1 or (dist[pair_v[v]] == dist[u] + 1 and dfs(pair_v[v])):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = float('inf')
        return False

    matching = 0
    while bfs():
        for u in range(n_left):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    return matching

def _can_match(m, L, n=12):
    # Build adjacency list
    # Left nodes: 0..n-1 corresponding to i=1..n
    # Right nodes: integers in (m, m+L]
    # We need to map integer values to indices
    
    # To speed up, we can just use the values directly if we use a dict for pair_v,
    # but standard HK uses indices.
    # The number of candidates is small (<= 144 * 12 / avg_i).
    # Let's collect all unique candidates.
    
    candidates = set()
    adj = [[] for _ in range(n)]
    
    for i in range(1, n + 1):
        # First multiple of i > m
        r = ((m // i) + 1) * i
        v = r
        end = m + L
        while v <= end:
            adj[i - 1].append(v)
            candidates.add(v)
            v += i
        if not adj[i - 1]:
            return False
            
    if not candidates:
        return False
        
    # Map candidates to indices
    cand_list = sorted(candidates)
    val_to_idx = {v: i for i, v in enumerate(cand_list)}
    n_right = len(cand_list)
    
    # Convert adj to index-based
    adj_idx = [[val_to_idx[v] for v in adj[i]] for i in range(n)]
    
    # Run Hopcroft-Karp
    max_matching = _hopcroft_karp(adj_idx, n, n_right)
    
    return max_matching == n

def _f(m, n=12):
    # Binary search smallest L in [1, 144]
    # We know L is at most 144 from the problem description
    lo, hi = 1, 144
    if not _can_match(m, hi, n):
        return hi
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
    
    # Generate critical m in [0, p-1]
    # m is critical if m+1 is a multiple of some i in 1..12
    critical_ms = set()
    for i in range(1, 13):
        for k in range(1, p // i + 1):
            m_val = k * i - 1
            if 0 <= m_val < p:
                critical_ms.add(m_val)
            
    # Also include 0 just in case (though 0 is critical as 1 is multiple of 1)
    critical_ms.add(0)
    
    # Sort for deterministic order
    critical_list = sorted(critical_ms)
    
    best_m = 0
    best_f = _f(0)
    
    for m in critical_list:
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m

    return int(best_m)
