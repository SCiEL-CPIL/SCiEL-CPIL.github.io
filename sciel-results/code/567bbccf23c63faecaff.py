import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Define Grötzsch Graph (Mycielski of C5)
    # Vertices 0..10
    groetzsch_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [5, 0], [5, 2],                          # Copy 5 of 0
        [6, 1], [6, 3],                          # Copy 6 of 1
        [7, 2], [7, 4],                          # Copy 7 of 2
        [8, 3], [8, 0],                          # Copy 8 of 3
        [9, 4], [9, 1],                          # Copy 9 of 4
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9] # Top 10
    ]
    
    # Initialize adjacency bitmasks
    # 16 vertices
    N = 16
    best_adj = [0] * N
    best_edge_count = 0
    
    # Base graph edges
    base_edges = groetzsch_edges[:]
    base_adj = [0] * N
    for u, v in base_edges:
        base_adj[u] |= (1 << v)
        base_adj[v] |= (1 << u)
    base_edge_count = len(base_edges)
    
    # Pre-compute all possible edges not in base graph
    # u < v
    all_possible_edges = []
    for u in range(N):
        for v in range(u + 1, N):
            if not (base_adj[u] & (1 << v)):
                all_possible_edges.append((u, v))
                
    # Shuffle for randomization in each iteration
    # We will use a seed or just random.shuffle
    
    while time.time() < deadline:
        # Reset to base
        adj = base_adj[:]
        edge_count = base_edge_count
        
        # Shuffle the candidate edges
        candidates = all_possible_edges[:]
        random.shuffle(candidates)
        
        # Greedy addition
        for u, v in candidates:
            # Check if adding (u,v) creates a triangle
            # Triangle exists if (u,v) shares a neighbor.
            # i.e., adj[u] & adj[v] != 0
            if (adj[u] & adj[v]) == 0:
                # Safe to add
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edge_count += 1
        
        # Update best
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            best_adj = adj[:]
            
    # Convert best_adj to list of edges
    result_edges = []
    for u in range(N):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if u < v:
                result_edges.append([u, v])
            mask ^= lsb
            
    return result_edges
