def construct():
    best_adj = None
    best_n = 0
    best_alpha = float('inf')
    
    # Step 1: Try to find a good n=39 graph
    adj_39 = None
    for seed in range(50):
        if _remaining_time() < 10:
            break
        result = _try_construction_bitmask(39, seed)
        if result is not None:
            adj_res, alpha_res = result
            if alpha_res < best_alpha:
                best_adj = adj_res
                best_n = 39
                best_alpha = alpha_res
                adj_39 = adj_res
            if alpha_res < 10:
                break

    if adj_39 is None:
        # Fallback to standard search if 39 failed
        for n_target in range(38, 20, -1):
            if _remaining_time() < 5:
                break
            for seed in range(20):
                if _remaining_time() < 2:
                    break
                result = _try_construction_bitmask(n_target, seed)
                if result is not None:
                    adj_res, alpha_res = result
                    if alpha_res < best_alpha:
                        best_adj = adj_res
                        best_n = n_target
                        best_alpha = alpha_res
                    if alpha_res < 10:
                        break
            if best_alpha < 10:
                break

    # Step 2: Try to extend to 40
    if best_n == 39 and _remaining_time() > 10:
        rng = random.Random(12345)
        adj_40 = _try_extend_to_40(best_adj, rng)
        if adj_40 is not None:
            best_adj = adj_40
            best_n = 40
            best_alpha = 9

    # Step 3: If 40 not found via extension, try random search for 40
    if best_n < 40 and _remaining_time() > 10:
        for seed in range(50):
            if _remaining_time() < 5:
                break
            result = _try_construction_bitmask(40, seed)
            if result is not None:
                adj_res, alpha_res = result
                if alpha_res < 10:
                    best_adj = adj_res
                    best_n = 40
                    best_alpha = alpha_res
                    break

    # Convert to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
