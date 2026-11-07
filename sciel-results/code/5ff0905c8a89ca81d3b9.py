    # --- Phase 1: Clarke-Wright Savings Construction ---
    # Initial routes: each customer is a separate route
    routes = [[c] for c in customers]
    # Map node to its current route index
    route_idx_of_node = {c: i for i, c in enumerate(customers)}
    
    # Calculate savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            c_i = customers[i]
            c_j = customers[j]
            s = dist[0, c_i] + dist[0, c_j] - dist[c_i, c_j]
            savings.append((s, c_i, c_j))
            
    savings.sort(reverse=True)
    
    for s, c_i, c_j in savings:
        if time.time() > deadline - 10.0:
            break
        if s < 0:
            break
            
        r_i = route_idx_of_node.get(c_i)
        r_j = route_idx_of_node.get(c_j)
        
        if r_i is None or r_j is None or r_i == r_j:
            continue
            
        route_i = routes[r_i]
        route_j = routes[r_j]
        
        # Check if we can merge
        # We can merge if c_i is at the end of route_i and c_j is at the start of route_j
        # OR if c_j is at the end of route_j and c_i is at the start of route_i
        
        merge_candidate = None
        if route_i[-1] == c_i and route_j[0] == c_j:
            merge_candidate = route_i + route_j
        elif route_j[-1] == c_j and route_i[0] == c_i:
            merge_candidate = route_j + route_i
            
        if merge_candidate:
            # Check feasibility
            feasible, _, _ = _check_route(merge_candidate, instance)
            if feasible:
                # Perform merge
                # Remove route_j and route_i
                # We need to be careful with indices. 
                # Let's just rebuild the routes list? No, that's O(N^2).
                # Better: mark as removed and append new.
                
                # Update mapping for nodes in merge_candidate
                new_route_idx = len(routes)
                for node in merge_candidate:
                    route_idx_of_node[node] = new_route_idx
                    
                # Remove old routes
                # To avoid index shifting issues, we can just append the new route and ignore the old ones 
                # by checking if they are still "active" in the mapping? 
                # No, `routes` list will contain stale routes. 
                # We should filter out stale routes at the end of the loop or maintain a set of active indices.
                
                # Simplest robust way for small N: 
                # 1. Create new route
                # 2. Remove old routes from list
                # 3. Append new route
                # 4. Update indices for ALL nodes in remaining routes? No, just the merged ones.
                # But removing shifts indices of subsequent routes.
                
                # Let's use a different structure: 
                # `routes` is a list of lists.
                # `route_idx_of_node` maps node -> index in `routes`.
                
                # If we remove r_j (higher index) first, then r_i, the indices of routes after r_j don't change for r_i removal?
                # If r_i < r_j:
                #   Remove r_j. Indices > r_j shift down by 1.
                #   Remove r_i. Indices > r_i shift down by 1.
                #   This is messy.
                
                # Alternative: Don't remove immediately. Just mark them.
                # Or, since N is small (25), we can just:
                # 1. Find the two routes.
                # 2. Merge them.
                # 3. Rebuild the `routes` list and `route_idx_of_node` map? 
                #    Rebuilding is O(N). Doing it for each merge (up to 24 merges) is O(N^2) = 625 ops. Very fast.
                
                # Let's do that. It's safer.
                
                new_route = merge_candidate
                # Remove old routes
                routes.remove(route_j)
                routes.remove(route_i)
                routes.append(new_route)
                
                # Rebuild map
                route_idx_of_node = {}
                for idx, r in enumerate(routes):
                    for node in r:
                        route_idx_of_node[node] = idx
                        
    # Rebuild map one last time
    route_idx_of_node = {}
    for idx, r in enumerate(routes):
        for node in r:
            route_idx_of_node[node] = idx
