"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    import time
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # Precompute validity of all subsets
    # valid[mask] = True if mask is a clique or an independent set
    valid = [False] * (1 << n)
    valid[0] = True  # empty set is trivially valid (though we won't use it as a part)
    
    for mask in range(1, 1 << n):
        # Check if clique
        is_c = True
        is_i = True
        m = mask
        while m:
            bit = m & (-m)
            u = bit.bit_length() - 1
            m ^= bit
            rest = mask ^ bit
            if rest & ~adj_mask[u]:
                is_c = False
            if rest & adj_mask[u]:
                is_i = False
            if not is_c and not is_i:
                break
        valid[mask] = is_c or is_i

    # DP: min_parts[mask] = minimum number of valid parts to cover 'mask'
    # We want min_parts[full_mask]
    # Transition: min_parts[mask] = 1 + min over valid submasks S of mask (where S contains the lowest set bit of mask) of min_parts[mask ^ S]
    # To ensure we don't count permutations, we fix that the part containing the lowest vertex is S.
    
    INF = 10**9
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    
    # Also store the best split for reconstruction
    best_split = [0] * (1 << n)
    
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lsb = mask & (-mask)
        
        # Iterate over all submasks of mask that contain lsb
        # Standard submask enumeration: sub = (sub - 1) & mask
        # We need sub to contain lsb.
        # Trick: iterate over submasks of (mask ^ lsb), then add lsb.
        rest = mask ^ lsb
        sub = rest
        while True:
            candidate = sub | lsb
            if valid[candidate]:
                remaining = mask ^ candidate
                if min_parts[remaining] + 1 < min_parts[mask]:
                    min_parts[mask] = min_parts[remaining] + 1
                    best_split[mask] = candidate
            
            if sub == 0:
                break
            sub = (sub - 1) & rest

    # Reconstruct the partition
    result = []
    current_mask = full_mask
    while current_mask:
        part_mask = best_split[current_mask]
        if part_mask == 0:
            # Fallback if something went wrong (shouldn't happen if valid[full_mask] is True or DP is correct)
            # This implies no valid split was found, which means min_parts[full_mask] is INF.
            # But single vertices are always valid (cliques of size 1), so it should always be solvable.
            part_mask = current_mask & (-current_mask)
        
        # Convert part_mask to list of vertices
        vertices = []
        m = part_mask
        while m:
            bit = m & (-m)
            v = bit.bit_length() - 1
            vertices.append(v)
            m ^= bit
        result.append(vertices)
        
        current_mask ^= part_mask

    return result

# EVOLVE-BLOCK-END
