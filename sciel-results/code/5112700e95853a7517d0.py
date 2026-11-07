"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = P['n']
    edges = P['edges']
    min_vertices = P['minimum_vertices']
    
    # Precompute adjacency bitmasks for fast degree calculation within subsets
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_score = -1
    best_subset = None
    
    # Enumerate all subsets of size at least min_vertices
    # Since n=17, total subsets is 2^17 = 131072. This is very fast.
    # We can iterate through all masks from 0 to 2^n - 1 and check popcount.
    # Or use itertools.combinations for each size.
    
    # Using combinations is cleaner and avoids checking popcount for every mask.
    # Sizes range from 8 to 17.
    
    for size in range(min_vertices, n + 1):
        if time.time() > deadline:
            break
            
        for subset in itertools.combinations(range(n), size):
            if time.time() > deadline:
                break
                
            # Calculate degrees of vertices in the induced subgraph
            # A vertex v in subset has degree equal to the number of neighbors in subset.
            # We can compute this using bit operations.
            mask = 0
            for v in subset:
                mask |= (1 << v)
                
            degrees = []
            for v in subset:
                # Count bits in (adj[v] & mask)
                deg = bin(adj[v] & mask).count('1')
                degrees.append(deg)
                
            distinct_degrees = len(set(degrees))
            
            if distinct_degrees > best_score:
                best_score = distinct_degrees
                best_subset = subset
                
                # The maximum possible distinct degrees in a graph of size 'size' is 'size'.
                # If we hit the theoretical maximum for the largest size (17), we can stop.
                # But we want the global max. The max possible is 17.
                # If best_score == size, we can't do better for this size.
                # If best_score == n, we can't do better overall.
                if best_score == n:
                    return list(best_subset)

    if best_subset is None:
        # Fallback to a valid subset if something went wrong (should not happen)
        return list(range(min_vertices))
        
    return list(best_subset)
# EVOLVE-BLOCK-END
