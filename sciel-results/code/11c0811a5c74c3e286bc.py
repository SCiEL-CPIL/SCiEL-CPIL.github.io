"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency as bitmasks for fast intersection
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = None
    best_distinct = 0
    
    deadline = time.time() + 85
    
    for size in range(min_v, n + 1):
        for combo in combinations(range(n), size):
            if time.time() > deadline:
                break
            # Build bitmask for this subset
            mask = 0
            for v in combo:
                mask |= (1 << v)
            # Compute degrees using bit operations
            degrees = set()
            for v in combo:
                deg = bin(adj_mask[v] & mask).count('1')
                degrees.add(deg)
            ndistinct = len(degrees)
            if ndistinct > best_distinct:
                best_distinct = ndistinct
                best_subset = combo
        if time.time() > deadline:
            break
    
    if best_subset is None:
        return list(range(min_v))
    
    return list(best_subset)
# EVOLVE-BLOCK-END
