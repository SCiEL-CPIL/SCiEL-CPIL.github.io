"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e), max(e)))
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Candidate edges: all non-existing, non-self edges
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u, v) not in base_edges:
                candidates.append((u, v))
    
    # Check if adding edge (u,v) creates a triangle
    def creates_triangle(u, v, adj):
        # Check if any common neighbor
        common = adj[u] & adj[v]
        return len(common) > 0
    
    # BFS to compute diameter
    def compute_diameter(adj):
        from collections import deque
        max_dist = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        max_dist = max(max_dist, dist[v])
                        q.append(v)
        return max_dist
    
    # Start with base graph
    current_adj = [set(s) for s in adj]
    
    # Greedy: repeatedly add the edge that most reduces the diameter
    # But we need to be careful about triangles
    
    # Actually, let's try a different approach: 
    # Add edges in a structured way to achieve diameter 4
    
    # Strategy: add edges to create a "ladder" with jumps
    # For diameter 4, we need that from any vertex, we can reach any other in <= 4 steps
    
    # Let's try adding edges (i, i+k) for various k, ensuring no triangles
    
    # A simple approach: add edges that skip 6 positions
    # (i, i+6) for i in 0..17, but check triangles
    
    added = []
    for i in range(n):
        j = i + 6
        if j < n and (i, j) not in base_edges and not creates_triangle(i, j, current_adj):
            current_adj[i].add(j)
            current_adj[j].add(i)
            added.append([i, j])
    
    # Check diameter
    diam = compute_diameter(current_adj)
    
    if diam > 4:
        # Need more edges, try adding more
        # Add edges (i, i+3) for remaining
        for i in range(n):
            j = i + 3
            if j < n and (i, j) not in base_edges and (i, j) not in [(a,b) for a,b in added] and not creates_triangle(i, j, current_adj):
                current_adj[i].add(j)
                current_adj[j].add(i)
                added.append([i, j])
        
        diam = compute_diameter(current_adj)
    
    if diam > 4:
        # Add more edges as needed
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in base_edges and (i, j) not in [(a,b) for a,b in added] and not creates_triangle(i, j, current_adj):
                    current_adj[i].add(j)
                    current_adj[j].add(i)
                    added.append([i, j])
                    if compute_diameter(current_adj) <= 4:
                        break
            if compute_diameter(current_adj) <= 4:
                break
    
    # Verify
    final_adj = [set() for _ in range(n)]
    for u, v in base_edges:
        final_adj[u].add(v)
        final_adj[v].add(u)
    for u, v in added:
        final_adj[u].add(v)
        final_adj[v].add(u)
    
    # Check triangle-free
    for u in range(n):
        for v in final_adj[u]:
            for w in final_adj[v]:
                if w != u and w in final_adj[u]:
                    # Triangle u-v-w
                    pass  # This shouldn't happen
    
    return added

# EVOLVE-BLOCK-END
