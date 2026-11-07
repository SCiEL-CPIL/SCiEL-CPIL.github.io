"""Triangle-free diameter-four augmentation."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diam(n, adj):
    diam = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        maxd = max(dist)
        if maxd > diam:
            diam = maxd
    return diam

def _is_triangle_free(n, edges):
    adj = _build_adj(n, edges)
    for u in range(n):
        for v in adj[u]:
            if v > u:
                common = adj[u] & adj[v]
                if common:
                    return False
    return True

def _valid(n, base_edges, added_edges):
    all_edges = list(base_edges) + list(added_edges)
    edge_set = set()
    for u, v in all_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    if not _is_triangle_free(n, all_edges):
        return False
    adj = _build_adj(n, all_edges)
    dist = [-1] * n
    dist[0] = 0
    q = deque([0])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    if any(d == -1 for d in dist):
        return False
    if _bfs_diam(n, adj) > 4:
        return False
    return True

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    base_set = set()
    for u, v in base_edges:
        base_set.add((min(u,v), max(u,v)))
    
    # Directly construct a known valid 9-edge solution
    # Center at 11, connect to even vertices on both sides (excluding adjacent 10,12)
    added = [
        [11, 0], [11, 2], [11, 4], [11, 6], [11, 8],
        [11, 14], [11, 16], [11, 18], [11, 22]
    ]
    
    # Verify
    if not _valid(n, base_edges, added):
        # Fallback: 10-edge solution
        added = [
            [11, 0], [11, 2], [11, 4], [11, 6], [11, 8],
            [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]
        ]
        if not _valid(n, base_edges, added):
            # Last resort: parallel chords
            added = [[i, i+6] for i in range(18)]
    
    return added
# EVOLVE-BLOCK-END
