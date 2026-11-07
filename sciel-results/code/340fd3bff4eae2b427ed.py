import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Build base adjacency
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))
    
    def bfs_max_dist(adj):
        """Compute eccentricity of all nodes (max distance from each node)."""
        ecc = [0] * n
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
            ecc[start] = max(dist)
        return max(ecc)
    
    def is_valid_add(u, v, adj, edge_set):
        """Check if adding edge (u,v) keeps graph triangle-free and doesn't duplicate."""
        if u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        # Triangle-free check: no common neighbor
        common = adj[u] & adj[v]
        return len(common) == 0
    
    # Greedy: repeatedly add the edge that most reduces diameter
    # until diameter <= 4
    
    # First, check if base graph already has diameter <= 4
    current_diam = bfs_max_dist(adj)
    if current_diam <= 4:
        return []
    
    added_edges = []
    deadline = time.time() + 85  # Safety margin
    
    # Generate all candidate edges (non-adjacent pairs)
    all_candidates = []
    for u in range(n):
        for v in range(u + 1, n):
            key = (u, v)
            if key not in edge_set:
                all_candidates.append((u, v))
    
    # Greedy loop
    while time.time() < deadline:
        current_diam = bfs_max_dist(adj)
        if current_diam <= 4:
            break
        
        best_edge = None
        best_new_diam = current_diam + 1  # We want to minimize
        
        # Try each candidate edge
        for u, v in all_candidates:
            if not is_valid_add(u, v, adj, edge_set):
                continue
            
            # Temporarily add edge
            adj[u].add(v)
            adj[v].add(u)
            key = (min(u, v), max(u, v))
            edge_set.add(key)
            
            new_diam = bfs_max_dist(adj)
            
            # Remove edge
            adj[u].remove(v)
            adj[v].remove(u)
            edge_set.remove(key)
            
            if new_diam < best_new_diam:
                best_new_diam = new_diam
                best_edge = (u, v)
        
        if best_edge is None or best_new_diam >= current_diam:
            # No improvement possible
            break
        
        u, v = best_edge
        adj[u].add(v)
        adj[v].add(u)
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        added_edges.append([u, v])
        
        # Remove from candidates
        all_candidates = [(a, b) for a, b in all_candidates if (a, b) != (u, v) and (a, b) != (v, u)]
    
    return added_edges
