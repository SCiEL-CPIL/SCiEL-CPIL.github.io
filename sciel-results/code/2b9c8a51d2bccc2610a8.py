import time
import random

def construct():
    # Start with the Grötzsch graph (11 vertices, 20 edges)
    # Vertices 0..10 are part of the graph, 11..15 are isolated initially
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    initial_edges = set()
    for u, w in cycle:
        initial_edges.add((min(u, w), max(u, w)))
    for u, w in cycle:
        initial_edges.add((min(u, 5 + w), max(u, 5 + w)))
        initial_edges.add((min(w, 5 + u), max(w, 5 + u)))
    for u in range(5):
        initial_edges.add((min(5 + u, 10), max(5 + u, 10)))
    
    # Convert to list for return later, but use set for logic
    # Note: The task asks for a list of edges [u, v]
    
    n = 16
    all_possible_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in initial_edges:
                all_possible_edges.append((i, j))
    
    best_edges = list(initial_edges)
    best_len = len(best_edges)
    
    start_time = time.time()
    max_time = 10.0 # Run for 10 seconds
    
    while time.time() - start_time < max_time:
        # Greedy randomized search
        current_edges = set(initial_edges)
        # Build adjacency list for current_edges
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        candidates = all_possible_edges[:]
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Check if adding (u, v) creates a triangle
            # This happens if there is a common neighbor w
            if not (adj[u] & adj[v]):
                # Safe to add
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        if len(current_edges) > best_len:
            best_len = len(current_edges)
            best_edges = list(current_edges)
            
    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
