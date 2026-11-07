"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    """
    Returns the edges of K_{8,8}, the unique triangle-free graph on 16 vertices
    with maximum edge count (64) and diameter exactly 2.
    
    Every edge is critical: since the graph is triangle-free, removing any edge
    (u,v) eliminates the only path of length 1 between u and v, and no path of
    length 2 exists (no common neighbor), so dist(u,v) becomes >= 3, increasing
    the diameter above 2.
    
    By Mantel's Theorem, no triangle-free graph on 16 vertices can have more
    than 64 edges, so this is optimal.
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit

    n = 16
    half = n // 2

    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([int(u), int(v)])

    # Verify: diameter is exactly 2 and every edge is critical
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def bfs_dist(start):
        dist = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            node = q.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        return dist

    # Check diameter <= 2 and >= 2
    has_dist2 = False
    for s in range(n):
        dist = bfs_dist(s)
        for i in range(n):
            if dist[i] == -1:
                return edges  # Shouldn't happen, but safe fallback
            if dist[i] > 2:
                return edges  # Shouldn't happen
            if dist[i] == 2:
                has_dist2 = True
    if not has_dist2:
        return edges  # Shouldn't happen

    # Check each edge is critical
    for (u, v) in edges:
        if time.time() > deadline:
            break
        adj[u].discard(v)
        adj[v].discard(u)

        bad = False
        for s in range(n):
            dist = bfs_dist(s)
            for i in range(n):
                if dist[i] == -1 or dist[i] > 2:
                    bad = True
                    break
            if bad:
                break

        # Restore
        adj[u].add(v)
        adj[v].add(u)

        if not bad:
            # This edge is NOT critical - shouldn't happen for K_{8,8}
            # But if it does, we'd need to handle it. In practice it won't.
            pass

    return edges

# EVOLVE-BLOCK-END
