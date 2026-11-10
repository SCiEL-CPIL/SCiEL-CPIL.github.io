def construct():
    n = 20
    best_edges = 0
    best_graph = []
    
    # Precompute all pairs
    pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Define multiple deterministic orderings
    # 1. Lexicographic
    order1 = pairs[:]
    # 2. Reverse Lexicographic
    order2 = pairs[::-1]
    # 3. Sorted by sum of indices
    order3 = sorted(pairs, key=lambda x: x[0] + x[1])
    # 4. Sorted by difference of indices
    order4 = sorted(pairs, key=lambda x: abs(x[0] - x[1]))
    # 5. Sorted by u, then v (standard)
    order5 = sorted(pairs, key=lambda x: (x[0], x[1]))
    # 6. Sorted by v, then u
    order6 = sorted(pairs, key=lambda x: (x[1], x[0]))
    
    # Generate more pseudo-random but deterministic orders using a simple LCG
    # This simulates "random restarts" but deterministically
    orders = [order1, order2, order3, order4, order5, order6]
    
    seed_val = 12345
    for i in range(50):
        # Simple LCG for shuffling
        a = 1103515245
        c = 12345
        m = 2**31
        # We will just create a deterministic permutation based on i
        # Using a simple sort with a key depending on i
        order_i = sorted(pairs, key=lambda x: (x[0] * (i+1) + x[1] * (i+2)) % (n*n))
        orders.append(order_i)
        
    for order in orders:
        adj = [0] * n  # Bitsets
        edges = []
        
        for u, v in order:
            # Check 1: Common neighbors of u and v
            # adj[u] & adj[v]
            common_uv = adj[u] & adj[v]
            if common_uv.bit_count() > 2:
                continue
            
            # Check 2 & 3: 
            # For all x in N(v), |N(u) & N(x)| <= 1
            # For all y in N(u), |N(v) & N(y)| <= 1
            
            valid = True
            # Iterate over neighbors of v
            mask_v = adj[v]
            while mask_v:
                lsb = mask_v & -mask_v
                x = lsb.bit_length() - 1
                mask_v ^= lsb
                common_ux = adj[u] & adj[x]
                if common_ux.bit_count() > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Iterate over neighbors of u
            mask_u = adj[u]
            while mask_u:
                lsb = mask_u & -mask_u
                y = lsb.bit_length() - 1
                mask_u ^= lsb
                common_vy = adj[v] & adj[y]
                if common_vy.bit_count() > 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Add edge
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            edges.append([u, v])
            
        if len(edges) > best_edges:
            best_edges = len(edges)
            best_graph = edges[:]
            
            if best_edges >= 48:
                break
                
    return best_graph
