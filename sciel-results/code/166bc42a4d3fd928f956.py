# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _popcount(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c

def construct():
    deadline = time.time() + 80.0
    n = P['n']
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    min_verts = P['minimum_vertices']  # 8
    
    best_degrees = 0
    best_subset_mask = 0
    
    # Precompute list of vertices
    vertices = list(range(n))
    
    total = 1 << n
    # We'll iterate over all subsets; skip those with popcount < min_verts
    # To be efficient, iterate and check popcount
    for mask in range(total):
        if time.time() > deadline:
            break
        # popcount of mask
        if _popcount(mask) < min_verts:
            continue
        # Compute degrees of each vertex in mask
        deg_set = set()
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            d = _popcount(adj[v] & mask)
            deg_set.add(d)
            m ^= lsb
        dc = len(deg_set)
        if dc > best_degrees:
            best_degrees = dc
            best_subset_mask = mask
            # Early termination: max possible distinct degrees for a subset of size k is k
            # but we can't easily bound; if we hit a high value, we can continue to be safe.
            # The theoretical max distinct degrees overall is bounded by n (17). 
            # If best_degrees == 17 we could stop, but unlikely.
    
    # Convert best_subset_mask to list of vertices
    result = []
    m = best_subset_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result = [int(x) for x in result]
    return result
# EVOLVE-BLOCK-END
