"""Rewritten program using bitmask-based exact search for cochromatic partition."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    size = 1 << n
    
    # Precompute is_clique and is_independent for all masks
    # is_clique[mask] is True if mask is a clique
    # is_indep[mask] is True if mask is an independent set
    is_clique = [False] * size
    is_indep = [False] * size
    is_valid = [False] * size
    
    for mask in range(1, size):
        lsb = mask & -mask
        u = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if rest == 0:
            is_clique[mask] = True
            is_indep[mask] = True
        else:
            # Check if rest is clique and u is connected to all in rest
            if is_clique[rest] and (rest & ~adj[u]) == 0:
                is_clique[mask] = True
            
            # Check if rest is indep and u is not connected to any in rest
            if is_indep[rest] and (rest & adj[u]) == 0:
                is_indep[mask] = True
        
        if is_clique[mask] or is_indep[mask]:
            is_valid[mask] = True
    
    # DP table
    # dp[mask] = min parts to cover mask
    dp = [10**9] * size
    dp[0] = 0
    
    # To reconstruct, we store the chosen submask
    choice = [0] * size
    
    for mask in range(1, size):
        best_val = 10**9
        best_sub = 0
        
        # Iterate over submasks of mask
        sub = mask
        while sub:
            if is_valid[sub]:
                rest = mask ^ sub
                val = 1 + dp[rest]
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        choice[mask] = best_sub
    
    # Reconstruct partition
    partition = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        if sub == 0:
            # Fallback if no solution found (should not happen)
            verts = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            partition.append(verts)
            break
        
        # Extract vertices
        verts = []
        m = sub
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        
        partition.append(verts)
        mask ^= sub
    
    return partition

# EVOLVE-BLOCK-END
