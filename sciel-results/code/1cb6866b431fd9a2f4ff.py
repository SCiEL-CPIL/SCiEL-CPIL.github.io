import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 20
    # Pre-compute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid(adj, u, v):
        # Check if adding edge (u, v) creates a K2,3
        # This happens if u and v share 3 or more common neighbors.
        # Common neighbors are w such that adj[u][w] == 1 and adj[v][w] == 1.
        common = 0
        for w in range(n):
            if w != u and w != v and adj[u][w] == 1 and adj[v][w] == 1:
                common += 1
                if common >= 3:
                    return False
        return True

    def add_edge(adj, u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    def count_edges(adj):
        # Since adj is symmetric, count edges where u < v
        return sum(1 for u in range(n) for v in range(u + 1, n) if adj[u][v] == 1)

    best_edges = 0
    best_adj = [[0] * n for _ in range(n)]
    
    # We will run multiple iterations with different random seeds/orderings
    # Each iteration: shuffle edges, try to add them greedily.
    
    num_iterations = 0
    while time.time() < deadline:
        random.shuffle(all_edges)
        adj = [[0] * n for _ in range(n)]
        
        for u, v in all_edges:
            if is_valid(adj, u, v):
                add_edge(adj, u, v)
        
        current_edges = count_edges(adj)
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj = [row[:] for row in adj]
            # Optional: if we hit a high threshold, we could stop early, but let's try to maximize
            # The seed is 30. K2,3-free graphs are related to polarity graphs or similar.
            # For n=20, the maximum number of edges is likely significantly higher than 30.
            # A random greedy approach usually finds decent dense subgraphs.
        
        num_iterations += 1
        # If we've done many iterations, we might want to break to ensure we return in time
        if num_iterations > 5000:
            break

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u][v]:
                edges.append([u, v])
                
    return edges
