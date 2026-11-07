"""Optimized exhaustive search to maximize distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=np.int32)
    for a, b in edges:
        adj[a, b] = 1
        adj[b, a] = 1
        
    best_count = 0
    best_vertices = list(range(minv))
    
    # Exhaustive search over all subsets of size >= minv
    # Total subsets: sum(C(17, k) for k in 8..17) ~ 64k
    # This is very fast, so we can afford to be thorough.
    
    for size in range(n, minv - 1, -1):
        if time.time() > deadline:
            break
            
        for combo in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
                
            # Vectorized degree calculation
            # Extract submatrix
            sub_adj = adj[np.ix_(combo, combo)]
            # Degrees are row sums
            degrees = sub_adj.sum(axis=1)
            # Count distinct degrees
            distinct_degrees = len(np.unique(degrees))
            
            if distinct_degrees > best_count:
                best_count = distinct_degrees
                best_vertices = list(combo)
                
            # Early termination if we found the theoretical maximum for this size
            # Max distinct degrees is 'size' (degrees 0 to size-1)
            if best_count == size:
                # We can't do better than 'size' for a subset of size 'size'
                # But we are iterating sizes from large to small.
                # If we found size distinct degrees in a subset of size 'size', 
                # that's the max possible for that size.
                # However, a smaller subset might have fewer max distinct degrees.
                # We want the global max across all sizes >= 8.
                # Since we start from largest size, if we find 'size' distinct degrees,
                # we know the answer is at least 'size'.
                # The maximum possible distinct degrees for any subset of size k is k.
                # So if we find a subset of size 17 with 17 distinct degrees, that's global max.
                # If we find size 16 with 16, that's likely global max.
                # We can break early if we find the max possible for the current largest size?
                # No, a subset of size 17 could have 10 distinct degrees, while a subset of size 8 has 8.
                # But 17 > 8. So we want the highest number.
                # The highest possible number is 17.
                # If we find 17, we can stop.
                # If we find 16, we can stop only if we know no subset can have 17.
                # Since we are iterating all subsets, we must check all unless we find the absolute max.
                # The absolute max is n=17.
                if best_count == n:
                    return [int(v) for v in best_vertices]
                    
    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
