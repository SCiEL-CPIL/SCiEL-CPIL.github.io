import time
from collections import deque

def _bfs_diameter(adj):
    """Return the diameter of the graph given by adjacency lists.
    Returns float('inf') if the graph is disconnected."""
    n = len(adj)
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
        for d in dist:
            if d == -1:
                return float('inf')
            if d > diam:
                diam = d
    return diam

def construct():
    """
    Return edges of a simple undirected graph on vertices 0..15 with diameter exactly 2,
    where deleting any single edge increases the diameter above 2 or disconnects the graph.
    
    The optimal solution is K_{8,8} (64 edges), which is triangle-free.
    By Mantel's Theorem, no triangle-free graph on 16 vertices has more than 64 edges.
    Since the criticality condition implies the graph must be triangle-free
    (for every edge (u,v), N(u) ∩ N(v) must be empty to ensure dist(u,v) >= 3 upon removal),
    64 is the maximum possible edge count.
    """
    n = 16
    half = 8
    result = []
    for u in range(half):
        for v in range(half, n):
            result.append([u, v])
    return result
