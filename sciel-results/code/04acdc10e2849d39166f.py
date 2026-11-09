import time

def construct():
    P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}
    
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Precompute adjacency masks
    adj_mask = [0] * n
    for a, b in P['edges']:
        adj_mask[a] |= (1 << b)
        adj_mask[b] |= (1 << a)
        
    # Precompute is_clique and is_indep for all masks
    size = 1 << n
    is_clique = [False] * size
    is_indep = [False] * size
    
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, size):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (adj_mask[v] & rest) == rest:
            is_clique[mask] = True
            
        if is_indep[rest] and (adj_mask[v] & rest) == 0:
            is_indep[mask] = True
            
    is_valid = [is_clique[i] or is_indep[i] for i in range(size)]
    
    # DP to find minimum parts
    # dp[mask] = min parts to partition mask
    INF = n + 1
    dp = [INF] * size
    dp[0] = 0
    # Store the choice to reconstruct the partition
    # choice[mask] = submask chosen as one part
    choice = [0] * size
    
    for mask in range(1, size):
        lsb = mask & -mask
        rest_mask = mask ^ lsb
        
        # Iterate over submasks of rest_mask
        # sub = lsb | sub_rest
        sub_rest = rest_mask
        while True:
            sub = sub_rest | lsb
            if is_valid[sub]:
                # Check if using this sub improves dp[mask]
                # The remaining part is mask ^ sub
                rem = mask ^ sub
                if 1 + dp[rem] < dp[mask]:
                    dp[mask] = 1 + dp[rem]
                    choice[mask] = sub
            if sub_rest == 0:
                break
            sub_rest = (sub_rest - 1) & rest_mask
            
    # Reconstruct partition
    mask = full_mask
    partition = []
    while mask:
        sub = choice[mask]
        # Convert sub to list of vertices
        verts = []
        for v in range(n):
            if (sub >> v) & 1:
                verts.append(v)
        partition.append(verts)
        mask ^= sub
        
    # Verify
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
        
    for part in partition:
        is_clique_part = True
        is_indep_part = True
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if (part[i], part[j]) in edge_set:
                    is_indep_part = False
                else:
                    is_clique_part = False
        if not (is_clique_part or is_indep_part):
            raise ValueError("Invalid partition found")
            
    return [list(map(int, part)) for part in partition]
