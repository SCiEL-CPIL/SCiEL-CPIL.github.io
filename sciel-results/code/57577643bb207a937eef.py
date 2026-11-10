"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

def bfs_diameter(adj, n):
    """Return the diameter of the graph given adjacency list.
    If graph is disconnected, return -1."""
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
        if -1 in dist:
            return -1
        d = max(dist)
        if d > diam:
            diam = d
    return diam

def check_validity(edges, n):
    """Check if edges form a valid graph with diameter exactly 2 and every edge is essential.
    Returns True if valid, False otherwise."""
    m = len(edges)
    # Build adjacency
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        a, b = (u, v) if u < v else (v, u)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj[a].append(b)
        adj[b].append(a)
    
    # Check diameter is exactly 2
    if bfs_diameter(adj, n) != 2:
        return False
    
    # Check each edge is essential (removing it disconnects or increases diameter above 2)
    for i, (u, v) in enumerate(edges):
        adj2 = [lst[:] for lst in adj]
        adj2[u].remove(v)
        adj2[v].remove(u)
        if bfs_diameter(adj2, n) <= 2:
            return False
    return True

def construct():
    """Construct a dense graph on 16 vertices with diameter exactly 2 where every edge is essential."""
    # Use the complete bipartite graph K_{8,8}.
    # Partition vertices 0..7 into set A and 8..15 into set B.
    # Edges exist only between A and B.
    # Diameter is 2.
    # Removing any edge (u,v) increases dist(u,v) to 3, so diameter becomes 3.
    # Thus every edge is essential.
    # Edge count: 8 * 8 = 64.
    
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([int(u), int(v)])
            
    return edges
# EVOLVE-BLOCK-END
