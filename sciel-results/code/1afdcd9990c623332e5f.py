"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks for fast induced subgraph degree calculation
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_score = -1
    best_vertices = None
    deadline = time.time() + 85.0
    
    # We want to maximize the number of distinct degrees.
    # The maximum possible distinct degrees for a subset of size k is k.
    # We search all subsets of size >= 8.
    # Since the total number of subsets is small (~90k), exhaustive search is feasible.
    
    for k in range(n, 7, -1):
        for combo in itertools.combinations(range(n), k):
            if time.time() > deadline:
                # Safety break, though unlikely to trigger
                break
            
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            # Calculate degrees in the induced subgraph
            # Using a list to avoid set overhead if possible, but set is fine for small k
            degrees = set()
            for v in combo:
                deg = (adj_mask[v] & mask).bit_count()
                degrees.add(deg)
            
            score = len(degrees)
            if score > best_score:
                best_score = score
                best_vertices = list(combo)
                
        if time.time() > deadline:
            break
            
    # Fallback if no valid combination was found (should not happen)
    if best_vertices is None:
        best_vertices = list(range(8))
        
    # Ensure return type is a list of Python ints
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
