import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # The known maximum n for R(K2,2,2) > n is 29.
    # R(K2,2,2) is known to be 31, so n=29 is the max valid coloring size.
    # We construct the Paley graph for n=29.
    
    n = 29
    
    # Quadratic residues mod 29
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
        
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            if diff in residues:
                colors.append(0)
            else:
                colors.append(1)
                
    # Build adjacency bitmasks
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
            
    # Fast verification for K2,2,2
    # K2,2,2 exists if there is a non-edge (u, v) such that the induced subgraph
    # on N(u) intersection N(v) contains a K2,2.
    # K2,2 exists in a graph H if there are adjacent vertices x, y in H 
    # with at least 2 common neighbors in H.
    
    def has_k222(adj_list, n):
        # Check color 0 (adj)
        for u in range(n):
            for v in range(u + 1, n):
                # Check if (u, v) is a non-edge in color 0
                if (adj_list[u] >> v) & 1:
                    continue
                cn_mask = adj_list[u] & adj_list[v]
                # Remove u and v from common neighbors (they are not neighbors of themselves)
                # But u and v are not in adj_list[u] or adj_list[v] anyway (no self loops)
                # However, u is not in adj_list[v] if (u,v) is non-edge.
                # v is not in adj_list[u] if (u,v) is non-edge.
                # So cn_mask is correct.
                
                # Iterate over edges (x, y) in the induced subgraph on cn_mask
                # We need to check if any such edge has >= 2 common neighbors in cn_mask
                m = cn_mask
                while m:
                    lsb_x = m & (-m)
                    x = lsb_x.bit_length() - 1
                    m ^= lsb_x
                    
                    # Neighbors of x in cn_mask
                    nx_mask = adj_list[x] & cn_mask
                    
                    # Iterate over neighbors y of x in cn_mask
                    my = nx_mask
                    while my:
                        lsb_y = my & (-my)
                        y = lsb_y.bit_length() - 1
                        my ^= lsb_y
                        
                        # Common neighbors of x and y in cn_mask
                        common = nx_mask & adj_list[y] & cn_mask
                        if common.bit_count() >= 2:
                            return True
        return False

    if has_k222(adj, n):
        # Fallback (should not happen for Paley 29)
        return {'n': 17, 'colors': [0] * (17*16//2)}

    # Check complement (color 1)
    comp_adj = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full ^ adj[i] ^ (1 << i)
        
    if has_k222(comp_adj, n):
        # Fallback
        return {'n': 17, 'colors': [0] * (17*16//2)}

    return {'n': n, 'colors': colors}
