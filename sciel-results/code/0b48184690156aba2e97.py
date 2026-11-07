import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    edges = P['edges']
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0

    # Build adjacency bitmask
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    best_count = 0
    best_mask = 0

    # Enumerate all masks with at least minv bits set
    # Total masks: 2^17 = 131072, which is fast
    for mask in range(1 << n):
        if time.time() > deadline:
            break
            
        # Check if mask has at least minv vertices
        if mask.bit_count() < minv:
            continue
            
        # Calculate degrees in induced subgraph
        deg_set = set()
        m = mask
        while m:
            # Get lowest set bit
            lowbit = m & (-m)
            v = lowbit.bit_length() - 1
            deg_set.add((adj[v] & mask).bit_count())
            m ^= lowbit
            
        current_count = len(deg_set)
        
        if current_count > best_count:
            best_count = current_count
            best_mask = mask
            # If we found k distinct degrees where k is the number of vertices,
            # this is the maximum possible for this subset size.
            # But a larger subset might have more distinct degrees.
            # If best_count equals the number of vertices in this mask, 
            # and that's the max possible (n), we can stop.
            if best_count == n:
                break
        elif current_count == best_count and current_count > 0:
            # Tie-breaking: prefer larger subsets (more vertices)
            if mask.bit_count() > best_mask.bit_count():
                best_mask = mask

    # Convert best_mask to list of vertex indices
    if best_mask == 0:
        # Fallback
        return [int(i) for i in range(minv)]
    
    result = []
    m = best_mask
    while m:
        lowbit = m & (-m)
        v = lowbit.bit_length() - 1
        result.append(int(v))
        m ^= lowbit
    
    return result
