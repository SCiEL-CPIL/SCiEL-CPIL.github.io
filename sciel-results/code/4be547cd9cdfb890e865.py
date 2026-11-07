"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def _bfs_distances(adj, src):
    """BFS from src, return list of distances."""
    n = len(adj)
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

def _get_diameter(adj):
    """Compute diameter of unweighted graph."""
    n = len(adj)
    max_d = 0
    for i in range(n):
        dists = _bfs_distances(adj, i)
        for d in dists:
            if d > max_d:
                max_d = d
    return max_d

def _build_adj(n, edges):
    """Build adjacency list from edge list."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def construct():
    start_time = time.time()
    
    n = 24
    # Base path edges
    base_edges = [(i, i + 1) for i in range(n - 1)]
    base_edge_set = set()
    for u, v in base_edges:
        base_edge_set.add((min(u,v), max(u,v)))
    
    # Candidate edges: all even-odd pairs not in base
    evens = list(range(0, n, 2))
    odds = list(range(1, n, 2))
    candidates = []
    for e in evens:
        for o in odds:
            key = (min(e,o), max(e,o))
            if key not in base_edge_set:
                candidates.append([e, o])
    
    # Greedy: repeatedly pick edge that most reduces diameter
    current_edges = list(base_edges)
    current_edge_set = set(base_edge_set)
    added = []
    
    # Compute initial diameter
    adj = _build_adj(n, current_edges)
    diam = _get_diameter(adj)
    
    max_iter = 50
    for _ in range(max_iter):
        if time.time() - start_time > 55:
            break
        if diam <= 4:
            break
        
        best_edge = None
        best_diam = diam
        
        for cand in candidates:
            u, v = cand
            key = (min(u,v), max(u,v))
            if key in current_edge_set:
                continue
            
            # Tentatively add edge
            new_edges = current_edges + [(u, v)]
            new_adj = _build_adj(n, new_edges)
            new_diam = _get_diameter(new_adj)
            
            if new_diam < best_diam:
                best_diam = new_diam
                best_edge = cand
        
        if best_edge is None or best_diam >= diam:
            break
        
        # Commit the edge
        u, v = best_edge
        key = (min(u,v), max(u,v))
        current_edges.append((u, v))
        current_edge_set.add(key)
        added.append([u, v])
        diam = best_diam
    
    if time.time() - start_time > 58:
        # Safety: return what we have
        return [[int(e[0]), int(e[1])] for e in added]
    
    return [[int(e[0]), int(e[1])] for e in added]

# EVOLVE-BLOCK-END
