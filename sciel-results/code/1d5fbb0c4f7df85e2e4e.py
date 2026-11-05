<<<<<<< SEARCH
def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try n from high to low
    for n_target in range(40, 20, -1):
        if _remaining_time() < 8:
            break
        
        found = False
        
        # Try random greedy + local search
        # Use multiple seeds
        for seed in range(100):
            if _remaining_time() < 3:
                break
            
            result = _try_construction_bitmask(n_target, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < best_alpha:
                    best_adj = adj_res
                    best_n = n_target
                    best_alpha = alpha_res
                if alpha_res < 10:
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        # Use a random greedy that we know works for small n
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
=======
def construct():
    """
    Construct a triangle-free graph on n vertices (1 <= n <= 41) with no independent set of 10.
    Maximize n.
    """
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Try n from high to low
    for n_target in range(40, 20, -1):
        if _remaining_time() < 6:
            break
        
        found = False
        
        # Try multiple seeds for this n
        for seed in range(50):
            if _remaining_time() < 3:
                break
            
            result = _try_construction_bitmask(n_target, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < best_alpha:
                    best_adj = adj_res
                    best_n = n_target
                    best_alpha = alpha_res
                if alpha_res < 10:
                    found = True
                    break
        
        if found:
            break
    
    # Fallback: use a known small construction if nothing worked
    if best_adj is None:
        n = 26
        rng = random.Random(42)
        adj = _greedy_triangle_free_bitmask(n, rng)
        best_adj = adj
        best_n = n
    
    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
>>>>>>> REPLACE
