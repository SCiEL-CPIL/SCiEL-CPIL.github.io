"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        u, v = e
        if u > v:
            u, v = v, u
        base_edges.add((u, v))
    
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    deadline = time.time() + 75.0
    
    def bfs_diameter(adj):
        diam = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            if any(d == -1 for d in dist):
                return float('inf')
            max_d = max(dist)
            if max_d > diam:
                diam = max_d
        return diam
    
    # Candidate edges: even-odd pairs not in base
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) in base_edges:
                continue
            if (u % 2) != (v % 2):
                candidates.append((u, v))
    
    cur_adj = [set(s) for s in adj]
    added = []
    added_set = set()
    cur_diam = bfs_diameter(cur_adj)
    
    while cur_diam > 4 and time.time() < deadline:
        best_edge = None
        best_diam = cur_diam
        for u, v in candidates:
            if (u, v) in added_set:
                continue
            cur_adj[u].add(v)
            cur_adj[v].add(u)
            d = bfs_diameter(cur_adj)
            cur_adj[u].remove(v)
            cur_adj[v].remove(u)
            if d < best_diam:
                best_diam = d
                best_edge = (u, v)
                if d <= 4:
                    break
        if best_edge is None:
            break
        u, v = best_edge
        cur_adj[u].add(v)
        cur_adj[v].add(u)
        added.append((u, v))
        added_set.add((u, v))
        cur_diam = best_diam
    
    result = [[u, v] for u, v in added]
    return result
# EVOLVE-BLOCK-END
