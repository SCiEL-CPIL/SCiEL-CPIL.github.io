import time
import random

def construct():
    # Start with the seed graph (Andrasfai-like, n=26)
    n = 26
    edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]
    
    # Helper to check if graph is valid
    def is_valid(n, edges):
        # Check for triangles
        adj = [[False]*n for _ in range(n)]
        for u, v in edges:
            adj[u][v] = True
            adj[v][u] = True
        
        # Triangle check
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        
        # Check for independent set of size 10
        # Use a simple branch and bound or heuristic for small n
        # For n up to 40, a basic B&B might be slow in Python, 
        # but we can use a greedy check or limit the search.
        # Since we are adding vertices, we can maintain the max independent set size.
        # However, for safety, let's use a simple recursive check with pruning.
        
        def find_independent_set_size(vertices, current_size):
            if len(vertices) == 0:
                return current_size
            if current_size + len(vertices) <= 9:
                return current_size
            
            # Pick a vertex
            v = vertices[0]
            rest = vertices[1:]
            
            # Option 1: Include v
            # Remove neighbors of v
            new_vertices = [u for u in rest if not adj[v][u]]
            size1 = find_independent_set_size(new_vertices, current_size + 1)
            
            # Option 2: Exclude v
            size2 = find_independent_set_size(rest, current_size)
            
            return max(size1, size2)
            
        max_alpha = find_independent_set_size(list(range(n)), 0)
        return max_alpha < 10

    # Try to extend the graph
    deadline = time.time() + 100  # 100 seconds deadline
    
    # We will try to add vertices from 26 up to 40
    for new_n in range(n + 1, 41):
        if time.time() > deadline:
            break
            
        # Try to add vertex new_n - 1
        # We need to find a set of edges to connect it to existing vertices
        # such that no triangle is formed and alpha < 10
        
        # Heuristic: Try to connect to non-neighbors of existing edges?
        # To avoid triangles, if we connect (u, v), u and v must not be connected.
        # So the new vertex can only connect to an independent set of the existing graph?
        # No. If we connect v_new to u, and u is connected to w, then v_new cannot connect to w.
        # So the neighbors of v_new must form an independent set in the existing graph.
        
        # Let's find all independent sets of the existing graph.
        # This is expensive.
        
        # Alternative: Randomly try subsets of edges for the new vertex.
        # For a new vertex, the set of neighbors must be an independent set.
        # So we can pick an independent set S from the existing graph, and connect v_new to all in S.
        # Then check if alpha increases to 10.
        
        # Let's find a few independent sets of the current graph.
        # Since alpha is small (<=9), we can find them.
        
        # Simple approach: Try to connect to a random subset of vertices, 
        # ensuring the subset is independent.
        
        found = False
        for _ in range(100): # 100 attempts per new vertex
            if time.time() > deadline:
                break
                
            # Pick a random independent set from the current graph
            # Greedy: pick random vertex, remove neighbors, repeat.
            remaining = list(range(new_n - 1))
            S = []
            while remaining:
                if random.random() < 0.5:
                    break
                v = random.choice(remaining)
                S.append(v)
                remaining = [u for u in remaining if u != v and not any((u, w) in edges or (w, u) in edges for w in S)]
            
            # Connect new vertex to S
            new_edges = edges + [[new_n - 1, u] for u in S]
            
            if is_valid(new_n, new_edges):
                edges = new_edges
                n = new_n
                found = True
                break
                
        if not found:
            # If we can't add a vertex, maybe we can modify edges?
            # For now, stop if we can't extend.
            break
            
    return {'n': n, 'edges': edges}
