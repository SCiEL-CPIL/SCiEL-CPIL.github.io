import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    n = 18
    # Adjacency matrix
    adj = [[0]*n for _ in range(n)]
    
    def is_k4_free():
        # Check if graph is K4-free
        # A K4 is a set of 4 vertices all connected to each other.
        # We can check all 4-subsets.
        # There are 3060 subsets.
        # For each, check 6 edges.
        # This is fast.
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if adj[u][v] and adj[u][w] and adj[u][x] and adj[v][w] and adj[v][x] and adj[w][x]:
                return False
        return True

    def get_max_triangle_free_size():
        # Find largest k such that there is a triangle-free induced subgraph of size k
        # Check k from n down to 1
        for k in range(n, 0, -1):
            for combo in itertools.combinations(range(n), k):
                # Check if combo is triangle-free
                # A triangle is a 3-clique.
                # Check all 3-subsets of combo
                has_triangle = False
                for tri in itertools.combinations(combo, 3):
                    a, b, c = tri
                    if adj[a][b] and adj[a][c] and adj[b][c]:
                        has_triangle = True
                        break
                if not has_triangle:
                    return k
        return 0

    def can_add_edge(u, v):
        if adj[u][v]:
            return False
        # Check if adding (u,v) creates a K4
        # This happens if there is an edge (w,x) in N(u) intersection N(v)
        # where w, x are distinct from u, v
        common_neighbors = []
        for i in range(n):
            if i != u and i != v and adj[u][i] and adj[v][i]:
                common_neighbors.append(i)
        
        # Check if any pair in common_neighbors has an edge
        for i in range(len(common_neighbors)):
            for j in range(i+1, len(common_neighbors)):
                w = common_neighbors[i]
                x = common_neighbors[j]
                if adj[w][x]:
                    return False
        return True

    # Initialize with empty graph
    # Or maybe start with a random sparse graph?
    # Empty graph has score 18.
    # Let's try to add random edges to break large triangle-free sets.
    
    best_graph = [row[:] for row in adj]
    best_score = n
    
    # Random restarts
    for restart in range(50):
        # Reset graph
        for i in range(n):
            for j in range(n):
                adj[i][j] = 0
        
        # Add random edges to make it "dense" but K4-free
        # We can try to add edges in a random order
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            if can_add_edge(u, v):
                adj[u][v] = 1
                adj[v][u] = 1
        
        current_score = get_max_triangle_free_size()
        
        if current_score < best_score:
            best_score = current_score
            best_graph = [row[:] for row in adj]
        
        # Local Search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_score = get_max_triangle_free_size()
            
            # Find a triangle-free subset of size current_score
            target_set = None
            for combo in itertools.combinations(range(n), current_score):
                has_triangle = False
                for tri in itertools.combinations(combo, 3):
                    a, b, c = tri
                    if adj[a][b] and adj[a][c] and adj[b][c]:
                        has_triangle = True
                        break
                if not has_triangle:
                    target_set = list(combo)
                    break
            
            if target_set is None:
                break # Should not happen
            
            # Try to add an edge within target_set
            # If we add an edge (u,v) in target_set, it might break the triangle-free property
            # specifically if it creates a triangle with another vertex in target_set
            # Or it might not change the max size if another set of same size exists.
            # But adding an edge can only reduce the size of the largest triangle-free set.
            
            # Try all pairs in target_set
            for i in range(len(target_set)):
                for j in range(i+1, len(target_set)):
                    u = target_set[i]
                    v = target_set[j]
                    if not adj[u][v] and can_add_edge(u, v):
                        # Add edge
                        adj[u][v] = 1
                        adj[v][u] = 1
                        
                        new_score = get_max_triangle_free_size()
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                            break
                        else:
                            # Revert
                            adj[u][v] = 0
                            adj[v][u] = 0
                if improved:
                    break
            
            if current_score < best_score:
                best_score = current_score
                best_graph = [row[:] for row in adj]

    # Convert best_graph to edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i][j]:
                edges.append([i, j])
                
    return edges
