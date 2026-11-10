"""Rewritten program using dynamic programming over subsets for optimal cochromatic partition."""

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    
    # Precompute adjacency bitmask
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute which subsets are cliques and which are independent sets
    # is_clique[S] = True if S induces a clique
    # is_indep[S] = True if S induces an independent set
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    
    # Empty set and singletons are both clique and independent
    is_clique[0] = True
    is_indep[0] = True
    for i in range(n):
        mask = 1 << i
        is_clique[mask] = True
        is_indep[mask] = True
        
    # Fill DP for clique/independent property
    # A set S is a clique if for all v in S, S \ {v} is a clique and v is connected to all in S \ {v}
    # Or simpler: S is clique iff for all u, v in S, (u,v) is an edge.
    # Efficient check: 
    # If S has one bit, it's a clique.
    # If S has multiple bits, let v be the lowest bit. S is clique iff (S without v) is clique AND v is connected to all in (S without v).
    # v connected to all in T iff (adj_mask[v] & T) == T.
    
    for mask in range(1, 1 << n):
        if mask & (mask - 1) == 0:
            continue # Single bit already handled
        
        # Get lowest set bit
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique
        if is_clique[rest]:
            # v must be adjacent to all in rest
            if (adj_mask[v] & rest) == rest:
                is_clique[mask] = True
        # If rest is not clique, mask cannot be clique.
        
        # Check independent
        if is_indep[rest]:
            # v must NOT be adjacent to any in rest
            if (adj_mask[v] & rest) == 0:
                is_indep[mask] = True
        # If rest is not independent, mask cannot be independent.

    # DP for minimum number of parts
    # dp[mask] = min parts to partition mask
    # choice[mask] = the subset T chosen as the first part
    dp = [float('inf')] * (1 << n)
    choice = [0] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        # Iterate all non-empty subsets T of mask
        # T is a valid part if is_clique[T] or is_indep[T]
        sub = mask
        while sub:
            if is_clique[sub] or is_indep[sub]:
                rest = mask ^ sub
                if dp[rest] + 1 < dp[mask]:
                    dp[mask] = dp[rest] + 1
                    choice[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct the partition
    full_mask = (1 << n) - 1
    parts = []
    current_mask = full_mask
    while current_mask:
        part_mask = choice[current_mask]
        # Convert mask to list of vertices
        part_vertices = []
        for i in range(n):
            if (part_mask >> i) & 1:
                part_vertices.append(i)
        parts.append(part_vertices)
        current_mask ^= part_mask
        
    # Convert to list of lists of ints
    result = [list(p) for p in parts]
    for part in result:
        for i in range(len(part)):
            part[i] = int(part[i])
            
    return result
