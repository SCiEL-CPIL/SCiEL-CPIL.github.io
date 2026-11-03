<<<<<<< SEARCH
    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    # Search downwards from best_scope - 1
    for scope in range(best_scope - 1, 99, -1):
        if time.time() - t0 > time_limit:
            break
            
        found = None
        # Try several seeds with varied node budgets
        # Use a higher budget for the first few seeds to find a solution
        for seed in range(12):
            if time.time() - t0 > time_limit:
                break
            # Increase budget for initial seeds
            if seed < 4:
                budget = 1000000
            else:
                budget = 500000
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
        
        if found:
            best_found = found
            best_scope_found = scope
        else:
            pass

    return best_found
=======
    t0 = time.time()
    time_limit = 115
    best_found = best_rows
    best_scope_found = best_scope
    
    # Search downwards from best_scope - 1
    lower_bound = 112
    for scope in range(best_scope - 1, lower_bound - 1, -1):
        if time.time() - t0 > time_limit:
            break
            
        found = None
        # Try several seeds with varied node budgets
        # Use a higher budget for the first few seeds to find a solution
        for seed in range(20):
            if time.time() - t0 > time_limit:
                break
            # Increase budget for initial seeds
            if seed < 5:
                budget = 2000000
            else:
                budget = 1000000
            res = solve_scope(scope, seed, max_nodes=budget)
            if res:
                found = res
                break
        
        if found:
            best_found = found
            best_scope_found = scope
        else:
            pass

    return best_found
>>>>>>> REPLACE
