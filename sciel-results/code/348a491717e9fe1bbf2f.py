"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    
    best_subset = None
    best_distinct_count = -1
    start_time = time.time()
    
    # Iterate over all possible subsets
    # 2^17 is small enough for exhaustive search within time limits
    for mask in range(1 << n):
        # Time check
        if time.time() - start_time > 45: # Safety margin
            break
            
        # Check size
        count = bin(mask).count('1')
        if count < min_verts:
            continue
        
        # Calculate degrees in induced subgraph
        degrees = []
        for v in range(n):
            if mask & (1 << v):
                # Count neighbors of v that are also in the subset
                deg = bin(mask & adj_masks[v]).count('1')
                degrees.append(deg)
        
        distinct_degrees = len(set(degrees))
        
        if distinct_degrees > best_distinct_count:
            best_distinct_count = distinct_degrees
            best_subset = mask
            
            # Early exit if we reach theoretical maximum?
            # Max distinct degrees is count (each vertex has a different degree)
            # But degrees are bounded by count-1. So max distinct is count.
            # Actually, max distinct degrees in a graph of k vertices is k (degrees 0 to k-1).
            # So if distinct_degrees == count, it's optimal for this size.
            # But a larger subset might have more distinct degrees?
            # Max distinct degrees for subset size k is k.
            # If we find a subset of size k with k distinct degrees, can a subset of size k+1 have k+1? Yes.
            # So we can't stop immediately unless we find a subset of size 17 with 17 distinct degrees.
            # Or if we know the global max.
            # Theoretical max distinct degrees for any induced subgraph of this graph is bounded by n.
            # But let's just run the full loop, it's fast.

    if best_subset is None:
        # Fallback to seed
        return list(range(min_verts))
    
    # Convert mask to list of vertices
    result = []
    for v in range(n):
        if best_subset & (1 << v):
            result.append(v)
            
    return result

# EVOLVE-BLOCK-END
