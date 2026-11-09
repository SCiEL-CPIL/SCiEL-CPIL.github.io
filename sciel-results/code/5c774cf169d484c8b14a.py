"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 16}
START_TIME = time.time()
TIME_LIMIT = 85

def bfs_distances(adj, start):
    n = len(adj)
    dist = [-1] * n
    dist[start] = 0
    q = deque([start])
    while q:
        u = q.popleft()
        du = dist[u]
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = du + 1
                q.append(v)
    return dist

def compute_diameter(adj):
    n = len(adj)
    max_dist = 0
    for i in range(n):
        dists = bfs_distances(adj, i)
        if -1 in dists:
            return -1
        max_dist = max(max_dist, max(dists))
    return max_dist

def is_diameter_two(adj):
    return compute_diameter(adj) == 2

def edge_is_critical(adj, u, v):
    # Remove edge
    try:
        adj[u].remove(v)
    except ValueError:
        pass
    try:
        adj[v].remove(u)
    except ValueError:
        pass
    
    new_diam = compute_diameter(adj)
    
    # Re-add edge
    adj[u].append(v)
    adj[v].append(u)
    
    return new_diam > 2 or new_diam == -1

def construct():
    n = P['n']
    
    # Strategy: Use Complete Bipartite Graph K_{8,8}
    # Partitions: 0-7 and 8-15
    # This graph has diameter 2.
    # Removing any edge (u, v) with u in A, v in B:
    # u and v are no longer adjacent.
    # Since the graph is bipartite, any path between u and v must have even length.
    # The shortest path was length 1. The next shortest possible is length 3.
    # Does a path of length 3 exist?
    # u connects to all B\{v}. v connects to all A\{u}.
    # Path u -> b' -> a' -> v where b' in B\{v}, a' in A\{u}.
    # Since |A|,|B| >= 2, such b' and a' exist.
    # So distance becomes 3.
    # Diameter becomes 3.
    # Thus, every edge is critical.
    
    partition_a = list(range(0, 8))
    partition_b = list(range(8, 16))
    
    edges = []
    for u in partition_a:
        for v in partition_b:
            edges.append([u, v])
            
    # Verification step (fast enough for n=16, 64 edges)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    if not is_diameter_two(adj):
        # Fallback: Star graph (diameter 2, all edges critical as bridges)
        return [[0, v] for v in range(1, n)]
        
    all_critical = True
    for u, v in edges:
        if not edge_is_critical(adj, u, v):
            all_critical = False
            break
            
    if not all_critical:
        # Fallback: Star graph
        return [[0, v] for v in range(1, n)]

    return edges

# EVOLVE-BLOCK-END
