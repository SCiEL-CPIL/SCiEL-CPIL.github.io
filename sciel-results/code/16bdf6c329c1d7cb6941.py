import time

def construct():
    n = 17
    minv = 8
    edges = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], 
             [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], 
             [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], 
             [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], 
             [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], 
             [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], 
             [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], 
             [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], 
             [8, 9], [8, 10], [8, 12], [8, 16], 
             [9, 10], [9, 11], [9, 13], 
             [10, 11], [10, 12], [10, 14], 
             [11, 12], [11, 13], [11, 15], 
             [12, 13], [12, 14], [12, 16], 
             [13, 14], [13, 15], 
             [14, 15], [14, 16], 
             [15, 16]]

    # Build adjacency masks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute: for each vertex, its adjacency mask
    # Also precompute a list of (bit_value, vertex_index) for each vertex
    # to speed up extraction

    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    best_count = -1
    best_mask = 0

    # The graph is 8-regular (each vertex has degree 8 in the full graph)
    # So in any induced subgraph, degrees range from 0 to min(8, sz-1)
    # Maximum possible distinct degrees is 9 (values 0 through 8)
    max_possible = 9

    for mask in range(1 << n):
        # Time guard with safety margin
        if time.time() > deadline:
            break

        sz = mask.bit_count()
        if sz < minv:
            continue

        # Quick upper bound: can't have more distinct degrees than sz
        # and can't exceed max_possible
        if sz + best_count < minv:  # This condition is always false, skip
            pass

        # Compute distinct degrees using a bitmask for speed
        deg_mask = 0  # bit d set means degree d was seen
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            deg_mask |= (1 << deg)
            m ^= lb

        nd = deg_mask.bit_count()
        if nd > best_count:
            best_count = nd
            best_mask = mask
            if best_count >= max_possible:
                break

    # Extract vertices from best_mask
    if best_mask == 0:
        return [int(i) for i in range(minv)]

    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb

    return verts
