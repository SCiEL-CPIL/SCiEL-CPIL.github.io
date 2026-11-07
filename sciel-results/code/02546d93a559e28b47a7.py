import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    best_distinct = 0
    best_mask = 0
    
    # Deadline guard
    deadline = time.time() + 50.0
    
    # Iterate over all possible subsets (masks)
    # We can optimize by iterating only masks with popcount >= minv
    # But checking popcount is cheap.
    # Total masks: 2^17 = 131072.
    
    for mask in range(1 << n):
        if time.time() > deadline:
            break
            
        # Quick check: if popcount < minv, skip
        # bit_count is a C-level function, very fast
        c = mask.bit_count()
        if c < minv:
            continue
            
        # Compute distinct degrees in the induced subgraph defined by mask
        deg_set = set()
        m = mask
        while m:
            # Get lowest set bit
            lb = m & (-m)
            # Vertex index
            v = lb.bit_length() - 1
            # Degree of v in the induced subgraph
            # Count neighbors of v that are also in mask
            d = (adj_mask[v] & mask).bit_count()
            deg_set.add(d)
            # Clear the bit
            m ^= lb
            
        distinct_count = len(deg_set)
        
        if distinct_count > best_distinct:
            best_distinct = distinct_count
            best_mask = mask
            
            # If we found the maximum possible distinct degrees for this size,
            # and we are iterating all masks, we can't get better than c.
            # But we don't know if a larger mask will have more.
            # Max possible is n.
            if best_distinct == n:
                break

    # Extract vertices from best_mask
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb
        
    # Ensure we return a list of ints
    return verts
