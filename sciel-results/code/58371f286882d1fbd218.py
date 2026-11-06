    # Precompute the 112 local Q3 masks on vertices 0..7
    local_edge_to_bit = {}
    k8_edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            local_edge_to_bit[(i, j)] = len(k8_edges)
            k8_edges.append((i, j))
            
    q3_local_masks = set()
    for p in permutations(range(8)):
        mask = 0
        for i in range(8):
            for j in range(i + 1, 8):
                if bin(i ^ j).count('1') == 1:
                    u, v = p[i], p[j]
                    if u > v: u, v = v, u
                    mask |= (1 << local_edge_to_bit[(u, v)])
        q3_local_masks.add(mask)
        
    q3_local_list = list(q3_local_masks)
    
    # Precompute global Q3 masks grouped by edge
    q3s_by_edge = [[] for _ in range(num_edges)]
    
    for subset in combinations(range(n), 8):
        # Build local (i,j) -> global_idx map
        # local i, j corresponds to subset[i], subset[j]
        # We need a fast way to get global index from local (i,j)
        # Let's create a 8x8 matrix or just a list of lists
        # local_map[i][j] = global_idx
        local_map = [[-1]*8 for _ in range(8)]
        for i in range(8):
            for j in range(i+1, 8):
                u, v = subset[i], subset[j]
                if u > v: u, v = v, u
                local_map[i][j] = edge_index[(u, v)]
                local_map[j][i] = local_map[i][j]
        
        for local_mask in q3_local_list:
            global_mask = 0
            m = local_mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                i, j = k8_edges[bit_idx]
                ge_idx = local_map[i][j]
                global_mask |= (1 << ge_idx)
                m ^= lsb
            
            gm = global_mask
            while gm:
                lsb = gm & -gm
                ge_idx = lsb.bit_length() - 1
                q3s_by_edge[ge_idx].append(global_mask)
                gm ^= lsb
