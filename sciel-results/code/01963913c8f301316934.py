import sys

def construct():
    # Graph definition from the prompt
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency masks
    # adj_mask[v] is a bitmask where bit u is 1 if (u, v) is an edge
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # non_adj_mask[v] is a bitmask where bit u is 1 if u != v and (u, v) is NOT an edge
    # Note: We only care about bits for u < n.
    all_mask = (1 << n) - 1
    non_adj_mask = [0] * n
    for v in range(n):
        # Start with all bits set, clear v's bit, clear neighbors' bits
        non_adj_mask[v] = all_mask & ~(1 << v) & ~adj_mask[v]

    # Precompute is_clique and is_independent for all subsets
    # Size 2^n
    size = 1 << n
    is_clique = [False] * size
    is_independent = [False] * size
    is_clique[0] = True
    is_independent[0] = True
    
    # valid[S] = is_clique[S] or is_independent[S]
    valid = [False] * size
    valid[0] = True
    
    for S in range(1, size):
        # Get lowest set bit
        lsb = S & -S
        v = lsb.bit_length() - 1
        R = S ^ lsb
        
        # Check clique
        # S is clique iff R is clique AND v is connected to all in R
        if is_clique[R]:
            # (adj_mask[v] & R) == R means all vertices in R are neighbors of v
            if (adj_mask[v] & R) == R:
                is_clique[S] = True
        else:
            is_clique[S] = False
            
        # Check independent
        # S is independent iff R is independent AND v is NOT connected to any in R
        if is_independent[R]:
            # (non_adj_mask[v] & R) == R means all vertices in R are non-neighbors of v
            if (non_adj_mask[v] & R) == R:
                is_independent[S] = True
        else:
            is_independent[S] = False
            
        valid[S] = is_clique[S] or is_independent[S]

    # DP for minimum set cover
    # dp[S] = min parts to cover subset S
    INF = float('inf')
    dp = [INF] * size
    dp[0] = 0
    parent = [-1] * size # parent[S] = the subset T chosen to cover part of S
    
    # Iterate through all subsets
    for S in range(1, size):
        # Iterate through all submasks T of S
        T = S
        while T > 0:
            if valid[T]:
                remaining = S ^ T
                cost = dp[remaining] + 1
                if cost < dp[S]:
                    dp[S] = cost
                    parent[S] = T
            T = (T - 1) & S
            
    # Reconstruct the partition
    parts = []
    S = size - 1 # All vertices
    while S > 0:
        T = parent[S]
        if T == -1:
            # This shouldn't happen if the graph is valid (single vertices are always valid)
            # But just in case, break
            break
        # Convert bitmask T to list of vertices
        subset_verts = []
        mask = T
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            subset_verts.append(v)
            mask ^= lsb
        parts.append(subset_verts)
        S = S ^ T
        
    return parts

# EVOLVE-BLOCK-END
