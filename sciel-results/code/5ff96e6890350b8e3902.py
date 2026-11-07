"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    deadline = time.time() + 80.0  # safety margin
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_score = -1
    best_mask = 0
    total_masks = 1 << n
    
    # Iterate through all possible subsets represented as bitmasks
    # We can optimize by only checking masks with popcount >= min_verts
    for mask in range(1, total_masks):
        if time.time() > deadline:
            break
            
        # Check if subset size is sufficient
        # bit_count is available in Python 3.8+
        size = mask.bit_count()
        if size < min_verts:
            continue
            
        # Compute distinct degrees
        degrees = set()
        # Iterate over set bits in mask
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            v = lsb.bit_length() - 1
            m -= lsb
            
            # Degree of v in induced subgraph
            deg = (adj_mask[v] & mask).bit_count()
            degrees.add(deg)
            
        score = len(degrees)
        if score > best_score:
            best_score = score
            best_mask = mask
            
    if best_mask == 0:
        # Fallback: construct a valid subset of size min_verts
        # Simple fallback: first min_verts vertices
        fallback_mask = (1 << min_verts) - 1
        return [i for i in range(n) if (fallback_mask >> i) & 1]
        
    # Convert best_mask back to list of vertices
    result = []
    m = best_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        result.append(v)
        m -= lsb
        
    return result

# EVOLVE-BLOCK-END
