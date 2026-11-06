import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n = 16
    
    # Grötzsch graph edges (11 vertices: 0..10)
    # Cycle 0-1-2-3-4-0
    # 5-0, 6-1, 7-2, 8-3, 9-4
    # 10-0, 10-1, 10-2, 10-3, 10-4
    # 5-6, 6-7, 7-8, 8-9, 9-5
    core_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (10, 0), (10, 1), (10, 2), (10, 3), (10, 4),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5)
    ]
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    core_set = set(core_edges)
    non_core_edges = [e for e in all_pairs if e not in core_set]
    
    best_edges = []
    best_count = 0
    
    for seed in range(10000): # Many restarts
        if time.time() > deadline:
            break
            
        random.seed(seed)
        adj = [0] * n
        for u, v in core_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        edge_count = len(core_edges)
        
        # Shuffle non-core edges
        shuffled = non_core_edges[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            # Check triangle-free
            if (adj[u] & adj[v]) == 0:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
                
        if edge_count > best_count:
            best_count = edge_count
            best_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if (adj[u] >> v) & 1:
                        best_edges.append([u, v])

    return best_edges
