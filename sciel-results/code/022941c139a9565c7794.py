"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_verts = P['minimum_vertices']
    
    # Build adjacency masks for fast bit-level degree computation
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_subset = None
    best_distinct = -1
    
    # Enumerate all subsets using bitmasks
    # 2^17 is small enough for a full search
    for mask in range(1 << n):
        # Check if number of vertices is at least min_verts
        # bin(mask).count('1') is efficient for small n
        if bin(mask).count('1') < min_verts:
            continue
        
        # Extract vertices
        vertices = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            vertices.append(v)
            m ^= lsb
        
        # Compute degrees in induced subgraph
        degrees = []
        for v in vertices:
            deg = bin(adj_mask[v] & mask).count('1')
            degrees.append(deg)
        
        distinct = len(set(degrees))
        if distinct > best_distinct:
            best_distinct = distinct
            best_subset = vertices
            
            # Optional: Early exit if we found the theoretical max for this size?
            # Max distinct degrees for k vertices is k.
            # If distinct == len(vertices), we can't do better for this size.
            # But we want global max. A larger subset might have same distinct count?
            # No, max distinct for size k is k.
            # If we find k distinct degrees in a subset of size k, that's optimal for that k.
            # But a subset of size k+1 can have k+1 distinct degrees.
            # So we can't stop globally unless we hit n distinct degrees.
            
    return best_subset
# EVOLVE-BLOCK-END
