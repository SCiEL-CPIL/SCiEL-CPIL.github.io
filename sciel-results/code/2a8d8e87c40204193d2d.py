"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.monotonic() + 80.0

    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    min_size = P['minimum_vertices']

    best_deg = 0
    best_vertices = list(range(min_size))
    
    # Iterate over all possible masks (subsets)
    # 2^17 is ~131k, which is very fast to iterate.
    for mask in range(1, 1 << n):
        if time.monotonic() > deadline:
            break
            
        size = mask.bit_count()
        if size < min_size:
            continue
            
        # Pruning: The number of distinct degrees cannot exceed the size of the subset.
        # If our current best is already >= size, this subset cannot improve the score.
        if best_deg >= size:
            continue
            
        # Calculate distinct degrees in the induced subgraph
        degs = set()
        m = mask
        # Optimization: if we already have 'size' distinct degrees, we can't get more.
        # But we need to compute them to know. 
        # We can break early if len(degs) == size? 
        # No, we still need to process bits to clear m? 
        # Actually, if len(degs) == size, we have found a "rainbow" degree set.
        # We can break the inner loop early? 
        # The set size is bounded by 'size'. If it reaches 'size', it's max for this mask.
        # We can just let it run, it's small.
        
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = (adj[v] & mask).bit_count()
            degs.add(d)
            if len(degs) == size:
                # Max possible distinct degrees for this size is 'size'.
                # We can break early to save time, as we know nd will be 'size'.
                # But we must ensure we don't miss anything? No, set size can't grow.
                break
            m -= lsb
            
        nd = len(degs)
        if nd > best_deg:
            best_deg = nd
            # Reconstruct vertex list from mask
            best_vertices = [i for i in range(n) if (mask >> i) & 1]
            
            # If we found a subset with 'size' distinct degrees, that is the max for this size.
            # However, larger sizes could have higher counts (e.g. 9 > 8).
            # So we continue unless best_deg == n (global max).
            if best_deg == n:
                break

    return [int(v) for v in best_vertices]
# EVOLVE-BLOCK-END
