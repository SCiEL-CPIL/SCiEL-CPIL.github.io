"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix or set for fast lookup
    # Using a set of frozensets or just a 2D boolean matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Precompute is_clique and is_indep for all masks
    # mask is an integer from 0 to 2^n - 1
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    is_clique[0] = True
    is_indep[0] = True
    
    # Helper to get lowest set bit
    # We can iterate masks and use the property:
    # Let v be the lowest set bit in mask.
    # rest = mask ^ (1 << v)
    # is_clique[mask] = is_clique[rest] and all(adj[v][u] for u in rest)
    # is_indep[mask] = is_indep[rest] and all(not adj[v][u] for u in rest)
    
    # To make this faster, we can precompute the "validity" based on the lowest bit.
    # However, iterating over all bits in 'rest' for every mask might be slow in Python if not careful.
    # 3^N is small, but the inner loop checks popcount(rest) bits.
    # Total operations approx sum_{mask} popcount(mask) * 2^(popcount(mask)-1) ? No.
    # Let's just do it. N=14 is tiny.
    
    for mask in range(1, size):
        # Find lowest set bit
        # v = (mask & -mask).bit_length() - 1
        lowbit = mask & -mask
        v = lowbit.bit_length() - 1
        rest = mask ^ lowbit
        
        # Check if v is connected to all in rest for clique
        # Check if v is NOT connected to any in rest for independent
        # We can iterate through bits of rest
        is_c = is_clique[rest]
        is_i = is_indep[rest]
        
        # Iterate bits in rest
        r = rest
        while r:
            lb = r & -r
            u = lb.bit_length() - 1
            if adj[v][u]:
                is_i = False # If edge exists, not independent
            else:
                is_c = False # If no edge, not clique
            r ^= lb
            
        is_clique[mask] = is_c
        is_indep[mask] = is_i
        
    # Precompute valid mask: clique OR independent
    is_valid = [False] * size
    for mask in range(size):
        is_valid[mask] = is_clique[mask] or is_indep[mask]
        
    # DP for minimum number of parts
    dp = [float('inf')] * size
    dp[0] = 0
    
    # parent[mask] stores the submask chosen to split mask
    parent = [0] * size
    
    # Iterate masks by increasing size (or just 1 to size-1 works if we ensure submasks are computed? 
    # Actually, dp[mask] depends on dp[mask ^ sub] where mask ^ sub < mask is NOT necessarily true numerically, 
    # but it is a proper subset, so it has fewer bits. 
    # If we iterate 1 to size-1, we don't guarantee that all subsets are computed.
    # We should iterate by popcount or ensure we check all submasks which are smaller in value? 
    # No, submask is always < mask in value? 
    # If sub is a submask of mask, then sub <= mask. 
    # If sub < mask, then dp[sub] is computed if we iterate 0..size-1.
    # Wait, mask ^ sub is a submask of mask. Is it always < mask? 
    # Yes, if sub is non-empty, mask ^ sub < mask.
    # So iterating mask from 1 to size-1 is correct.
    
    for mask in range(1, size):
        # Iterate all non-empty submasks of mask
        sub = mask
        best_val = float('inf')
        best_sub = 0
        while sub:
            if is_valid[sub]:
                val = 1 + dp[mask ^ sub]
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
            
        dp[mask] = best_val
        parent[mask] = best_sub
        
    # Reconstruct the partition
    full_mask = size - 1
    parts = []
    cur = full_mask
    while cur:
        sub = parent[cur]
        if sub == 0:
            # Should not happen if dp is finite
            break
        # Extract vertices from sub
        verts = []
        s = sub
        while s:
            lb = s & -s
            v = lb.bit_length() - 1
            verts.append(v)
            s ^= lb
        parts.append(verts)
        cur ^= sub
        
    # Ensure types are correct: list of lists of ints
    # The problem statement says: "sequences must be list or tuple"
    # "integers must be Python int"
    # My implementation uses lists of ints.
    
    return parts

# EVOLVE-BLOCK-END
