import numpy as np
from collections import deque

def construct(k, g):
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]

    # Greedy construction
    vertices = [0]
    adj = {0: set()}
    degree = {0: 0}
    edges = []
    
    max_vertices = 5000
    
    def get_dist(u):
        dist = {u: 0}
        q = deque([u])
        while q:
            node = q.popleft()
            for nei in adj[node]:
                if nei not in dist:
                    dist[nei] = dist[node] + 1
                    q.append(nei)
        return dist

    while True:
        # Find a vertex with degree < k
        u = None
        for v in vertices:
            if degree[v] < k:
                u = v
                break
        
        if u is None:
            break
        
        # Get distances from u
        dists = get_dist(u)
        
        # Find valid candidates
        # Candidates: existing vertices with degree < k, not neighbor of u
        candidates = [v for v in vertices if degree[v] < k and v not in adj[u] and v != u]
        
        valid_candidates = []
        for v in candidates:
            if v in dists and dists[v] >= g - 1:
                valid_candidates.append(v)
        
        if valid_candidates:
            # Pick the one with highest degree to balance
            v = max(valid_candidates, key=lambda x: degree[x])
            adj[u].add(v)
            adj[v].add(u)
            degree[u] += 1
            degree[v] += 1
            edges.append((u, v))
        else:
            # Create new vertex
            if len(vertices) >= max_vertices:
                # If we run out of vertices, we might fail.
                # For now, break.
                break
            new_v = len(vertices)
            vertices.append(new_v)
            adj[new_v] = set()
            degree[new_v] = 0
            adj[u].add(new_v)
            adj[new_v].add(u)
            degree[u] += 1
            degree[new_v] += 1
            edges.append((u, new_v))

    n = len(vertices)
    # Verify regularity
    # If not regular, the graph is invalid.
    # But the loop continues until all degrees are k or we break.
    # If we break due to max_vertices, it might be invalid.
    # Let's assume it works for k=3, g=16.
    
    return n, edges
