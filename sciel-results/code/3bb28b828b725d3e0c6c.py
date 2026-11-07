"""Combined program: exhaustive bitmask enumeration with bit-parallel degree computation."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 90s limit
    
    n = P['n']
    min_v = P['minimum_vertices']
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    best_score = 0
    best_mask = (1 << min_v) - 1  # Fallback: first min_v vertices
    
    # Precompute list of vertices for each mask popcount >= min_v
    # We iterate over all 2^17 masks and compute distinct degrees
    
    # Optimization: precompute the list of vertices in each mask using bit operations
    # For n=17, 2^17 = 131072 masks, each requiring at most 17 degree computations
    
    # Use a fast approach: for each mask, extract vertices and compute degrees
    # Key insight: we can use a lookup table or direct bit manipulation
    
    # Direct enumeration with early termination
    for mask in range(1 << n):
        # Time guard
        if time.time() > deadline:
            break
        
        c = mask.bit_count()
        if c < min_v:
            continue
        
        # Early exit: max distinct degrees for size c is c
        # If c <= best_score, we cannot improve
        if c <= best_score:
            continue
        
        # Compute distinct degrees for this mask
        degs = set()
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            d = (adj_mask[v] & mask).bit_count()
            degs.add(d)
            # Early termination: if we've found c distinct degrees, that's the max
            if len(degs) == c:
                break
            m ^= lb
        
        nd = len(degs)
        if nd > best_score:
            best_score = nd
            best_mask = mask
            
            # If we reached the theoretical maximum possible (n distinct degrees), stop
            if best_score == n:
                break
    
    # Build vertex list from best_mask
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb
    
    # Ensure we have at least min_v vertices (should always be true from search)
    if len(verts) < min_v:
        for i in range(n):
            if i not in verts:
                verts.append(i)
            if len(verts) >= min_v:
                break
    
    # Return as list of Python ints
    return [int(x) for x in verts]
# EVOLVE-BLOCK-END
