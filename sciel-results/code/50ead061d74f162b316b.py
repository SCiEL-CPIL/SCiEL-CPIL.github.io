import time
from collections import deque

def construct():
    t0 = time.time()
    deadline = 85.0
    n = 16

    # K(8,8) complete bipartite: 64 edges, all essential.
    # Mantel's Theorem: max edges in triangle-free graph on n vertices is n^2/4.
    # For n=16, that's 64. K(8,8) is the unique extremal example.
    # In K(8,8), every edge (u,v) with u in A, v in B has NO common neighbors
    # (A and B are disjoint, and within each part there are no edges).
    # Removing any edge (u,v) makes d(u,v) = 3 (must go through other vertices),
    # so diameter becomes 3 > 2. All edges are essential.
    
    A = list(range(0, 8))
    B = list(range(8, 16))
    edges = [[a, b] for a in A for b in B]

    # Fast verification:
    # 1. Diameter is exactly 2 (K(8,8) with both parts non-empty has diameter 2)
    # 2. Every edge is essential: in a triangle-free graph, edge (u,v) is essential
    #    iff u and v have no common neighbor (otherwise d(u,v)=2 via that neighbor)
    if _verify_fast(edges, n):
        return edges

    # Fallback: star graph (15 edges), diameter 2, all edges essential
    return [[0, v] for v in range(1, n)]

def _verify_fast(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    # Check diameter is exactly 2 using BFS
    def get_diameter():
        maxd = 0
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        q.append(w)
            if -1 in dist:
                return None  # Disconnected
            maxd = max(maxd, max(dist))
        return maxd

    d = get_diameter()
    if d != 2:
        return False

    # Check every edge is essential:
    # For a triangle-free graph, edge (u,v) is essential iff N(u) ∩ N(v) = ∅.
    for u, v in edges:
        if adj[u] & adj[v]:
            return False

    return True
