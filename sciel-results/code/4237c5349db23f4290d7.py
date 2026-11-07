"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin out of 60s

    n = P['n']
    edges = P['edges']
    
    # Precompute adjacency matrix for fast lookup
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True

    best_subset = list(range(8)) # Fallback
    max_distinct_degrees = 0

    # For each vertex, precompute its adjacency bitmask for faster degree counting
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # Iterate over all subset sizes from 8 to 17
    for k in range(8, n + 1):
        # Check deadline periodically (not every iteration)
        if time.time() > deadline:
            break
            
        count = 0
        for subset in itertools.combinations(range(n), k):
            # Check deadline every 1000 iterations to reduce overhead
            count += 1
            if count % 1000 == 0 and time.time() > deadline:
                return list(best_subset)
            
            # Calculate degrees in induced subgraph using bitmask
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            
            degrees = set()
            for v in subset:
                # Count neighbors of v within the subset
                deg = bin(adj_mask[v] & subset_mask).count('1')
                degrees.add(deg)
            
            distinct_count = len(degrees)
            if distinct_count > max_distinct_degrees:
                max_distinct_degrees = distinct_count
                best_subset = list(subset)
                
                # Early exit: if we found k distinct degrees, that's the maximum possible
                if distinct_count == k:
                    return list(best_subset)
                
    return list(best_subset)
# EVOLVE-BLOCK-END
