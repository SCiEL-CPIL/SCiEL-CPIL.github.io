"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    min_vertices = P['minimum_vertices']
    edges = P['edges']
    
    # Precompute adjacency masks
    adj_masks = [0] * n
    for u, v in edges:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
        
    best_score = -1
    best_subset_mask = 0
    
    # Iterate over all possible subsets
    # We only care about subsets with size >= min_vertices
    for mask in range(1 << n):
        # Quick check on size
        if mask.bit_count() < min_vertices:
            continue
            
        # Calculate degrees in induced subgraph
        degrees = set()
        # Iterate through vertices in the subset
        m = mask
        while m:
            # Get least significant set bit
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            
            # Degree is number of neighbors in the subset
            deg = (adj_masks[v] & mask).bit_count()
            degrees.add(deg)
            
        num_distinct_degrees = len(degrees)
        
        if num_distinct_degrees > best_score:
            best_score = num_distinct_degrees
            best_subset_mask = mask
            
    # Convert best_subset_mask to list of vertex indices
    result = []
    m = best_subset_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        m ^= lsb
        result.append(int(v))
        
    return result
# EVOLVE-BLOCK-END
