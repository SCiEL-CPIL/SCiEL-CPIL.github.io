import time
import sys
from collections import deque

def _dinic_max_flow(edges, n, source, sink):
    """
    Dinic's algorithm for max flow with sparse edges.
    edges: list of (u, v, cap) tuples
    Returns the max flow value.
    """
    # Build adjacency list with edge indices
    g = [[] for _ in range(n)]
    edge_u = []
    edge_v = []
    edge_cap = []
    
    def add_edge(u, v, cap):
        edge_u.append(u)
        edge_v.append(v)
        edge_cap.append(cap)
        g[u].append(len(edge_u) - 1)
        # Reverse edge
        edge_u.append(v)
        edge_v.append(u)
        edge_cap.append(0)
        g[v].append(len(edge_u) - 1)
    
    for u, v, cap in edges:
        add_edge(u, v, cap)
    
    flow = 0
    INF = 10**9
    
    while True:
        # BFS to find levels
        level = [-1] * n
        level[source] = 0
        q = deque([source])
        while q:
            u = q.popleft()
            for ei in g[u]:
                v = edge_v[ei]
                if edge_cap[ei] > 0 and level[v] < 0:
                    level[v] = level[u] + 1
                    q.append(v)
                    
        if level[sink] < 0:
            break
            
        # DFS to find blocking flow
        it = [0] * n
        
        def dfs(u, f):
            if u == sink:
                return f
            for i in range(it[u], len(g[u])):
                it[u] = i
                ei = g[u][i]
                v = edge_v[ei]
                if edge_cap[ei] > 0 and level[u] < level[v]:
                    d = dfs(v, min(f, edge_cap[ei]))
                    if d > 0:
                        edge_cap[ei] -= d
                        edge_cap[ei ^ 1] += d
                        return d
            return 0
            
        while True:
            f = dfs(source, INF)
            if f == 0:
                break
            flow += f
            
    return flow

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1...b_n with i | b_i.
    Uses Max Flow (Dinic) with sparse edges.
    """
    # Nodes:
    # 0: Source
    # 1..n: Indices i
    # n+1..n+L: Values v in (m, m+L]
    # n+L+1: Sink
    
    N = 1 + n + L + 1
    source = 0
    sink = N - 1
    
    edges = []
    
    # Source to indices
    for i in range(1, n + 1):
        edges.append((source, i, 1))
    
    # Indices to values
    # Value v corresponds to node index: n + (v - (m+1)) + 1
    for i in range(1, n + 1):
        for v in range(m + 1, m + L + 1):
            if v % i == 0:
                node_v = n + (v - (m + 1)) + 1
                edges.append((i, node_v, 1))
                
    # Values to sink
    for v in range(m + 1, m + L + 1):
        node_v = n + (v - (m + 1)) + 1
        edges.append((node_v, sink, 1))
    
    # Run Dinic
    max_flow = _dinic_max_flow(edges, N, source, sink)
    
    return max_flow == n

def _f_val(m, n=12):
    """
    Find the least positive L such that (m, m+L] contains a valid matching.
    """
    # Max L is stated to be at most 144
    for L in range(1, 145):
        if _can_match(m, L, n):
            return L
    return 144 # Fallback, shouldn't happen

def construct():
    """
    Find m in 0..1000000 that maximizes f(12, m).
    """
    # The function f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to search m in [0, 27720)
    K = 27720
    best_m = 0
    best_L = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds margin (limit is 90s)
    
    for m in range(K):
        if time.time() > deadline:
            break
        L = _f_val(m)
        if L > best_L:
            best_L = L
            best_m = m
            # If we hit the theoretical max, we can stop early
            if best_L == 144:
                break
                
    return int(best_m)
