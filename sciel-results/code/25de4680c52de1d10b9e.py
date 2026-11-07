import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    minv = P['minimum_vertices']
    edges = P['edges']
    
    # Build adjacency bitmask array
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)
        
    # Localize variables for speed
    adj_local = adj
    n_local = n
    minv_local = minv
    
    best_count = 0
    best_mask = 0
    
    # Precompute popcount for all masks
    # 2^17 = 131072, very fast
    popcount = [0] * (1 << n_local)
    for i in range(1, 1 << n_local):
        popcount[i] = popcount[i >> 1] + (i & 1)
        
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0 # Safety margin
    
    for mask in range(1, 1 << n_local):
        # Time check every 1024 iterations
        if (mask & 1023) == 0:
            if time.time() > deadline:
                break
                
        k = popcount[mask]
        if k < minv_local:
            continue
            
        # Calculate distinct degrees using integer bitmask
        # Max degree is 16, so 17 bits needed.
        deg_mask = 0
        m = mask
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            deg = (adj_local[v] & mask).bit_count()
            deg_mask |= (1 << deg)
            m ^= lb
            
        current_count = deg_mask.bit_count()
        
        if current_count > best_count:
            best_count = current_count
            best_mask = mask
            # Max possible distinct degrees is n (0 to n-1).
            if best_count == n_local:
                break

    # Extract vertices from best_mask
    verts = []
    m = best_mask
    while m:
        lb = m & (-m)
        v = lb.bit_length() - 1
        verts.append(int(v))
        m ^= lb
        
    return verts
# EVOLVE-BLOCK-END
