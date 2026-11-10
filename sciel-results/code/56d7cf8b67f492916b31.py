import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix as bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute clique and independent set status for all masks
    size = 1 << n
    is_clique = [False] * size
    is_ind = [False] * size
    is_valid = [False] * size
    
    is_clique[0] = True
    is_ind[0] = True
    is_valid[0] = True
    
    for mask in range(1, size):
        lb = mask & -mask
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        if is_clique[rest]:
            if (rest & ~adj[v]) == 0:
                is_clique[mask] = True
                
        if is_ind[rest]:
            if (rest & adj[v]) == 0:
                is_ind[mask] = True
                
        is_valid[mask] = is_clique[mask] or is_ind[mask]

    # DP for minimum number of parts
    # dp[mask] = min parts to cover mask
    # choice[mask] = the submask chosen as the part containing the lowest bit of mask
    INF = n + 1
    dp = [INF] * size
    choice = [0] * size
    dp[0] = 0
    
    # Iterate masks from 1 to size-1
    for mask in range(1, size):
        # Find lowest set bit
        lb = mask & -mask
        rest = mask ^ lb
        
        best_val = INF
        best_sub = 0
        
        # Iterate submasks of 'rest' to form 'sub = lb | sub_rest'
        # This ensures 'sub' contains the lowest bit of 'mask', breaking symmetry
        sub_rest = rest
        while True:
            sub = lb | sub_rest
            if is_valid[sub]:
                rem = mask ^ sub
                val = dp[rem] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest
        
        dp[mask] = best_val
        choice[mask] = best_sub

    # Reconstruct the partition
    parts = []
    current_mask = size - 1
    while current_mask:
        sub = choice[current_mask]
        if sub == 0:
            # Should not happen if valid solution exists, but safety check
            # If no valid sub found, it means mask itself is not valid?
            # But single vertex is always valid. So choice should never be 0 for non-zero mask.
            break
        # Convert sub to list of vertices
        verts = []
        for i in range(n):
            if sub & (1 << i):
                verts.append(i)
        parts.append(verts)
        current_mask ^= sub

    # Ensure types are correct
    return parts

# EVOLVE-BLOCK-END
