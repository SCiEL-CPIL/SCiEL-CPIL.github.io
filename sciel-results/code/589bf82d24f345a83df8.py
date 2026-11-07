"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_size = P['minimum_vertices']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    best_deg = 0
    best_vertices = None
    deadline = time.monotonic() + 70.0  # Safety margin before 90s kill

    # Precompute list of vertices for each possible mask size to speed up
    # Actually, let's just iterate efficiently.
    
    # Use a faster approach: for each mask, compute degrees using precomputed adj
    # Key optimization: avoid repeated time checks and use tighter inner loop
    
    for mask in range(1 << n):
        pc = mask.bit_count()
        if pc < min_size:
            continue
        
        # Compute distinct degrees in induced subgraph
        degs = set()
        m = mask
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            degs.add((adj[v] & mask).bit_count())
            m ^= low
            
        nd = len(degs)
        if nd > best_deg:
            best_deg = nd
            # Extract vertices
            verts = []
            m2 = mask
            while m2:
                low = m2 & (-m2)
                verts.append(low.bit_length() - 1)
                m2 ^= low
            best_vertices = verts
            # Early termination: if we found pc distinct degrees, 
            # that's the max possible for this size. But larger sizes 
            # could have more. Only stop if best_deg == n (impossible) 
            # or if we've reached a theoretical max.
            # For size k, max distinct degrees is k. If best_deg >= 8 
            # and we're at size 8, that's optimal for that size but 
            # larger sizes could do better.
            # Since 2^17 is small, just finish the search.
        
        # Periodic time check (every 4096 masks to reduce overhead)
        if (mask & 4095) == 4095 and time.monotonic() > deadline:
            break

    if best_vertices is None:
        best_vertices = list(range(min_size))
        
    return [int(v) for v in best_vertices]

# EVOLVE-BLOCK-END
