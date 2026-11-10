<<<<<<< SEARCH
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Start with a structured initialization: complete graph minus a matching
    # or use random dense starts
    
    # First try a structured start: complete bipartite-like
    # Split into two sets of 10, connect all across
    adj_struct = [0] * n
    for i in range(10):
        for j in range(10, 20):
            adj_struct[i] |= (1 << j)
            adj_struct[j] |= (1 << i)
    adj_struct = local_search(adj_struct, max_iters=200)
    cnt = edge_count(adj_struct)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = list(adj_struct)
    
    # Multiple random restarts
    for restart in range(150):
        if time.time() > deadline - 5:
            break
        
        density = 0.4 + rng.random() * 0.35
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        adj = local_search(adj, max_iters=150)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    # Final polish on the best solution
    if best_adj is not None:
        best_adj = polish(best_adj, max_iters=50)
        best_edges = edge_count(best_adj)
    
    if best_adj is None:
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
=======
    best_adj = None
    best_edges = 0
    rng = random.Random(42)
    
    # Start with multiple structured initializations
    # 1. Complete bipartite K_{10,10}
    adj_struct = [0] * n
    for i in range(10):
        for j in range(10, 20):
            adj_struct[i] |= (1 << j)
            adj_struct[j] |= (1 << i)
    adj_struct = local_search(adj_struct, max_iters=200)
    cnt = edge_count(adj_struct)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = list(adj_struct)
    
    # 2. Dense random start with high density
    adj_dense = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.7:
                adj_dense[i] |= (1 << j)
                adj_dense[j] |= (1 << i)
    adj_dense = local_search(adj_dense, max_iters=200)
    cnt = edge_count(adj_dense)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = list(adj_dense)
    
    # 3. Another structured: complete graph on 15 vertices plus random connections
    adj_struct2 = [0] * n
    for i in range(15):
        for j in range(i + 1, 15):
            adj_struct2[i] |= (1 << j)
            adj_struct2[j] |= (1 << i)
    for i in range(15):
        for j in range(15, 20):
            if rng.random() < 0.5:
                adj_struct2[i] |= (1 << j)
                adj_struct2[j] |= (1 << i)
    adj_struct2 = local_search(adj_struct2, max_iters=200)
    cnt = edge_count(adj_struct2)
    if cnt > best_edges:
        best_edges = cnt
        best_adj = list(adj_struct2)
    
    # Multiple random restarts with varied densities
    densities = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        
        density = densities[restart % len(densities)] + rng.random() * 0.1
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        adj = local_search(adj, max_iters=150)
        
        cnt = edge_count(adj)
        if cnt > best_edges:
            best_edges = cnt
            best_adj = list(adj)
    
    # Final polish on the best solution
    if best_adj is not None:
        best_adj = polish(best_adj, max_iters=50)
        best_edges = edge_count(best_adj)
    
    if best_adj is None:
        best_adj = [0] * n
        best_edges = 0

    edges = mask_to_edges(best_adj)
    return edges
>>>>>>> REPLACE
