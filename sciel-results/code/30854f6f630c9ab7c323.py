"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adjacency(n, base_edges, added_edges):
    adj = [[] for _ in range(n)]
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(n, adj, src):
    dist = [-1] * n
    dist[src] = 0
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

def _check_diameter(n, base_edges, added_edges):
    adj = _build_adjacency(n, base_edges, added_edges)
    max_dist = 0
    for i in range(n):
        dists = _bfs_distances(n, adj, i)
        for d in dists:
            if d > max_dist:
                max_dist = d
    return max_dist

def _is_bipartite_edge(n, base_edges, added_edges, u, v):
    # Check if adding (u,v) would create a triangle
    # Since base is bipartite, just check u and v are in different partitions
    # Partition: even vertices in A, odd in B
    return (u % 2) != (v % 2)

def construct():
    n = P['n']
    base_edges = P['edges']
    base_set = set()
    for e in base_edges:
        base_set.add(tuple(sorted(e)))
    
    # All candidate edges: between even and odd, not in base
    candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_set:
                candidates.append((u, v))
    
    # Greedy: start with no edges, add edges that reduce diameter most
    # Actually, start with a feasible solution and try to remove edges
    # First, find a feasible solution by adding all candidates
    added = list(candidates)
    
    # Try to remove edges one by one
    improved = True
    while improved:
        improved = False
        for i in range(len(added)):
            test_added = added[:i] + added[i+1:]
            if _check_diameter(n, base_edges, test_added) <= 4:
                added = test_added
                improved = True
                break
    
    return [[u, v] for u, v in added]
# EVOLVE-BLOCK-END
