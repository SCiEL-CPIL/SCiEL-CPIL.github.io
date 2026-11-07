"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(edges, n):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_distances(adj, src, n):
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

def _max_diameter(adj, n):
    maxd = 0
    for s in range(n):
        d = _bfs_distances(adj, s, n)
        maxd = max(maxd, max(d))
    return maxd

def _is_triangle_free(adj, u, v, n):
    # Check if adding edge (u,v) creates a triangle
    # Triangle exists if there's a common neighbor of u and v
    nu = set(adj[u])
    for w in adj[v]:
        if w in nu:
            return False
    return True

def _all_pairs_distances(adj, n):
    """Compute all-pairs shortest paths using BFS from each node."""
    dists = [[-1] * n for _ in range(n)]
    for s in range(n):
        dists[s] = _bfs_distances(adj, s, n)
    return dists

def construct():
    n = P['n']
    base_edges = [tuple(e) for e in P['edges']]
    norm_edges = set()
    for u, v in base_edges:
        norm_edges.add((min(u, v), max(u, v)))
    
    adj = _build_adj(base_edges, n)
    added = []
    max_iter = 20  # Should be enough for N=24
    
    for _ in range(max_iter):
        # Compute current diameter and distances
        dists = _all_pairs_distances(adj, n)
        diam = max(max(row) for row in dists)
        
        if diam <= 4:
            break
        
        # Generate candidates:
        # 1. Not in graph
        # 2. Triangle-free
        # 3. Current distance > 4 (pruning: short edges don't help reduce max diameter)
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u, v) in norm_edges:
                    continue
                if dists[u][v] <= 4:
                    continue
                if _is_triangle_free(adj, u, v, n):
                    candidates.append((u, v))
        
        if not candidates:
            break
            
        best_edge = None
        best_score = float('inf')
        
        # Evaluate candidates
        # Optimization: Sort by current distance descending? 
        # Or just iterate. N=24 is small.
        for u, v in candidates:
            # Tentatively add
            adj[u].append(v)
            adj[v].append(u)
            
            # Compute new diameter
            new_dists = _all_pairs_distances(adj, n)
            new_diam = max(max(row) for row in new_dists)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            
            if new_diam < best_score:
                best_score = new_diam
                best_edge = (u, v)
                if new_diam <= 4:
                    break
        
        if best_edge is None:
            break
            
        u, v = best_edge
        adj[u].append(v)
        adj[v].append(u)
        norm_edges.add((min(u, v), max(u, v)))
        added.append([u, v])
        
    # Local Search: Try to remove edges if diameter stays <= 4
    # This can reduce the count if greedy added redundant edges
    for i in range(len(added) - 1, -1, -1):
        u, v = added[i]
        # Remove edge
        adj[u].remove(v)
        adj[v].remove(u)
        norm_edges.discard((min(u, v), max(u, v)))
        
        # Check diameter
        dists = _all_pairs_distances(adj, n)
        diam = max(max(row) for row in dists)
        
        if diam <= 4:
            # Keep it removed
            added.pop(i)
        else:
            # Put it back
            adj[u].append(v)
            adj[v].append(u)
            norm_edges.add((min(u, v), max(u, v)))
            
    return [[int(u), int(v)] for u, v in added]
# EVOLVE-BLOCK-END
