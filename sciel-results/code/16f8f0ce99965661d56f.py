"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_graph(added_edges):
    """Build adjacency from base path + added edges."""
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _bfs_diameter(adj):
    """Compute diameter using BFS from all vertices."""
    n = len(adj)
    max_dist = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1  # disconnected
        max_dist = max(max_dist, max(dist))
    return max_dist

def _is_triangle_free(adj):
    """Check if graph is triangle-free."""
    n = len(adj)
    adj_set = [set(a) for a in adj]
    for u in range(n):
        for i, v in enumerate(adj[u]):
            for w in adj[u][i+1:]:
                if w in adj_set[v]:
                    return False
    return True

def _is_valid(added_edges):
    """Check if added edges produce a valid graph."""
    # Check for duplicate edges
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    adj = _build_graph(added_edges)
    if not _is_triangle_free(adj):
        return False
    diameter = _bfs_diameter(adj)
    if diameter < 0 or diameter > 4:
        return False
    return True

def _check_bipartite_edges(added_edges):
    """Verify all added edges are between even and odd vertices (bipartite)."""
    for u, v in added_edges:
        if (u % 2) == (v % 2):
            return False
    return True

def _fast_validate(added_edges):
    """Fast validation assuming bipartite edges (no triangle check needed)."""
    # Check duplicates
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
    for u, v in added_edges:
        key = (min(u,v), max(u,v))
        if key in edge_set:
            return False
        edge_set.add(key)
    
    # Build adjacency and check diameter
    n = 24
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    for u, v in added_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS diameter check
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return False
        if max(dist) > 4:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Phase 1: Try to find a 9-edge solution
    # Strategy: For each possible even hub vertex, try all subsets of 9 odd vertices
    # that include the "critical" odd vertices needed for diameter ≤ 4
    
    best_solution = None
    
    # First, establish a known valid 10-edge solution as fallback
    base_solution = [
        [12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
        [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]
    ]
    if _fast_validate(base_solution):
        best_solution = base_solution
    else:
        # Fallback: try to find any valid solution
        return []
    
    # Phase 2: Search for 9-edge solutions
    # Key insight: With a single hub at even vertex h, we need 9 odd vertices connected.
    # The odd vertices are {1,3,5,...,23}. We must exclude one.
    # For each excluded odd vertex o, check if the remaining 9 work.
    
    odd_vertices = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    # Try each even vertex as hub with 9 of 12 odd vertices
    even_vertices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    
    found_9 = False
    for h in even_vertices:
        if time.time() > deadline:
            break
        # Try excluding each odd vertex
        for exclude in odd_vertices:
            if time.time() > deadline:
                break
            candidate = [[h, o] for o in odd_vertices if o != exclude]
            if len(candidate) != 9:
                continue
            if _fast_validate(candidate):
                best_solution = candidate
                found_9 = True
                break
        if found_9:
            break
    
    # Phase 3: If no 9-edge solution with single hub, try two-hub configurations
    if not found_9:
        # Try configurations with two even hubs sharing the 9 odd connections
        # e.g., hub1 connects to k odds, hub2 connects to (9-k) odds, no overlap
        # This is more complex; try a few promising configurations
        for h1 in even_vertices:
            if time.time() > deadline:
                break
            for h2 in even_vertices:
                if h2 <= h1:
                    continue
                if time.time() > deadline:
                    break
                # Try splitting odd vertices between h1 and h2
                # h1 gets first k, h2 gets rest
                for k in range(1, 9):
                    if time.time() > deadline:
                        break
                    # h1 connects to odd_vertices[0:k], h2 to odd_vertices[k:k+9-k]
                    # But we need exactly 9 total, so h2 gets 9-k
                    if 9 - k > 12 - k:
                        continue
                    candidate = [[h1, o] for o in odd_vertices[:k]] + \
                                 [[h2, o] for o in odd_vertices[k:k+9-k]]
                    if len(candidate) != 9:
                        continue
                    if _fast_validate(candidate):
                        best_solution = candidate
                        found_9 = True
                        break
                if found_9:
                    break
            if found_9:
                break
    
    # Ensure return type is correct
    result = []
    for e in best_solution:
        result.append([int(e[0]), int(e[1])])
    
    return result
# EVOLVE-BLOCK-END
