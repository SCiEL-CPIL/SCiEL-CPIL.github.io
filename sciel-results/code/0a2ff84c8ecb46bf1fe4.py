"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency as bitmask for fast operations
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    all_vertices = list(range(n))
    best_score = -1
    best_subset = None
    
    # Enumerate subsets by size, from largest to smallest
    # For a subset of size k, max distinct degrees is k
    # We want to find the global maximum
    for size in range(n, min_verts - 1, -1):
        if time.time() > deadline:
            break
        for subset in combinations(all_vertices, size):
            if time.time() > deadline:
                break
            
            # Build bitmask for this subset
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            # Compute degrees in induced subgraph
            degrees = set()
            for v in subset:
                # Count neighbors of v that are in the subset
                neighbors_in_subset = adj_mask[v] & mask
                deg = bin(neighbors_in_subset).count('1')
                degrees.add(deg)
            
            distinct_count = len(degrees)
            
            if distinct_count > best_score:
                best_score = distinct_count
                best_subset = list(subset)
            
            # Early exit: if we found the theoretical max for this size,
            # no need to check more subsets of this size
            if distinct_count == size:
                break
        
        # If we found size distinct degrees, we can't do better with smaller subsets
        # (since max distinct degrees for size k is k)
        if best_score >= size:
            break
    
    if best_subset is None:
        return list(range(min_verts))
    
    return [int(v) for v in best_subset]
# EVOLVE-BLOCK-END
