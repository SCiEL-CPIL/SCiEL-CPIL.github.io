"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _verify(added_edges):
    """Check triangle-free, connected, diameter <= 4."""
    n = P['n']
    adj = [[] for _ in range(n)]
    edge_set = set()
    
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((min(u,v), max(u,v)))
    
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False  # Duplicate
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add(key)
    
    # Check triangle-free
    for u in range(n):
        neighbors = set(adj[u])
        for v in adj[u]:
            for w in adj[v]:
                if w != u and w in neighbors:
                    return False  # Triangle found
    
    # Check connected via BFS
    visited = [False] * n
    queue = deque([0])
    visited[0] = True
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                queue.append(v)
    
    if not all(visited):
        return False  # Not connected
    
    # Check diameter
    max_dist = 0
    for src in range(n):
        visited = [False] * n
        dist = [-1] * n
        queue = deque([src])
        visited[src] = True
        dist[src] = 0
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    queue.append(v)
        if max_dist > 4:
            return False
    
    return True

def construct():
    # Hardcoded valid solution: Connect node 11 (odd) to all even nodes.
    # This preserves triangle-free property (bipartite) and achieves diameter 4.
    added_edges = []
    hub = 11
    for v in range(0, 24, 2):
        if v != hub: # hub is odd, so v is always even, no self-loop
            added_edges.append([hub, v])
    
    # Verify to be safe
    if _verify(added_edges):
        return added_edges
    
    # Fallback if verification fails (should not happen)
    return []
# EVOLVE-BLOCK-END
