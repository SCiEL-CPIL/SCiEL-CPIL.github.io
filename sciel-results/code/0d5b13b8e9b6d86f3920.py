import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Grötzsch graph edges
    # Vertices 0-4: C5
    # Vertices 5-9: Copies
    # Vertex 10: Top
    edges = []
    # C5
    for i in range(5):
        edges.append([i, (i + 1) % 5])
    # Copies connected to neighbors
    for i in range(5):
        # i' is 5+i
        # Neighbors of i in C5 are (i-1)%5 and (i+1)%5
        n1 = (i - 1) % 5
        n2 = (i + 1) % 5
        edges.append([5 + i, n1])
        edges.append([5 + i, n2])
    # Top connected to copies
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Initial graph
    n = 16
    adj = [[0] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = 1
        adj[v][u] = 1
        
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True

    def is_3_colorable(adj):
        colors = [-1] * n
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                if all(colors[u] != c for u in range(n) if adj[v][u]):
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        return backtrack(0)

    def get_mis(adj):
        # Greedy MIS
        mis = []
        remaining = list(range(n))
        while remaining:
            # Pick vertex with min degree in remaining subgraph
            min_deg = float('inf')
            min_v = -1
            for v in remaining:
                deg = sum(adj[v][u] for u in remaining)
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
            mis.append(min_v)
            # Remove min_v and its neighbors
            neighbors = [u for u in remaining if adj[min_v][u]]
            remaining.remove(min_v)
            for u in neighbors:
                remaining.remove(u)
        return mis

    def add_vertex(adj, v, neighbors):
        for u in neighbors:
            adj[v][u] = 1
            adj[u][v] = 1
            # Add edge to global list if not present? 
            # We just need to return the final edge list.
            
    best_edges = edges[:]
    best_count = len(edges)
    
    # Try all permutations of vertices 11-15
    for perm in itertools.permutations([11, 12, 13, 14, 15]):
        if time.time() > deadline:
            break
            
        current_adj = [row[:] for row in adj]
        current_edges = edges[:]
        
        for v in perm:
            mis = get_mis(current_adj)
            # Connect v to all in mis
            for u in mis:
                current_adj[v][u] = 1
                current_adj[u][v] = 1
                current_edges.append([v, u])
                
        # Check if valid (triangle-free and not 3-colorable)
        # It should be triangle-free by construction.
        # It should be 4-chromatic because it contains Grötzsch.
        # But let's verify 3-colorability to be safe (it shouldn't be 3-colorable).
        if not is_3_colorable(current_adj):
            if len(current_edges) > best_count:
                best_count = len(current_edges)
                best_edges = current_edges[:]
                
    return best_edges
