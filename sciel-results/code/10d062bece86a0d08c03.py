<<<<<<< SEARCH
def _cluster_and_solve(instance, deadline):
    """Cluster customers and solve each cluster independently."""
    D = _build_dist_matrix(instance)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Sort by due time for better clustering
    customers.sort(key=lambda v: instance["due"][v])
    
    # Create clusters using nearest neighbor
    routes = []
    remaining = list(customers)
    
    while remaining:
        cluster = _nearest_neighbor_cluster(remaining, instance, D, max_nodes=4)
        if not cluster:
            break
        
        # Sort cluster by due time
        cluster.sort(key=lambda v: instance["due"][v])
        
        # Check if the cluster route is feasible
        feasible, dist = _check_route(cluster, instance, D)
        if feasible:
            routes.append(cluster)
            remaining = [c for c in remaining if c not in set(cluster)]
        else:
            # Try smaller cluster
            if len(cluster) > 1:
                cluster = cluster[:len(cluster)//2 + 1]
                cluster.sort(key=lambda v: instance["due"][v])
                feasible, dist = _check_route(cluster, instance, D)
                if feasible:
                    routes.append(cluster)
                    remaining = [c for c in remaining if c not in set(cluster)]
                else:
                    # Single customer
                    single = [cluster[0]]
                    feasible, dist = _check_route(single, instance, D)
                    if feasible:
                        routes.append(single)
                        remaining = [c for c in remaining if c != cluster[0]]
                    else:
                        # This shouldn't happen, but handle it
                        remaining.remove(cluster[0])
            else:
                # Single customer
                single = [cluster[0]]
                feasible, dist = _check_route(single, instance, D)
                if feasible:
                    routes.append(single)
                remaining = [c for c in remaining if c != cluster[0]]
    
    # Local search
    if time.time() < deadline - 5:
        routes = _local_search(routes, instance, D, max_iterations=20)
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 120  # Leave margin for safety
    
    # Build distance matrix
    D = _build_dist_matrix(instance)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Try clustering approach
    routes = _cluster_and_solve(instance, deadline)
    
    # Verify all customers are served
    served = set()
    for route in routes:
        for v in route:
            if v <= n_customers:
                served.add(v)
    
    # Add any missing customers as single routes
    for c in customers:
        if c not in served:
            routes.append([c])
    
    # Final verification
    valid_routes = []
    for route in routes:
        feasible, dist = _check_route(route, instance, D)
        if feasible:
            valid_routes.append(route)
    
    return valid_routes
# EVOLVE-BLOCK-END
=======
def _savings_construction(instance, D):
    """Construct initial routes using Clarke-Wright Savings heuristic."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Initial routes: each customer is its own route
    routes = [[c] for c in customers]
    
    # Calculate savings: s(i, j) = D[0][i] + D[0][j] - D[i][j]
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s = D[0][i] + D[0][j] - D[i][j]
                savings.append((s, i, j))
    
    # Sort savings in descending order
    savings.sort(reverse=True)
    
    # Map customer to route index
    route_of_customer = {c: i for i, route in enumerate(routes) for c in route}
    
    for s, i, j in savings:
        if s <= 0:
            continue
        
        ri = route_of_customer.get(i)
        rj = route_of_customer.get(j)
        
        if ri is None or rj is None or ri == rj:
            continue
        
        # Check if i and j are at the ends of their respective routes
        route_i = routes[ri]
        route_j = routes[rj]
        
        # i must be at start or end of route_i
        if i not in (route_i[0], route_i[-1]):
            continue
        # j must be at start or end of route_j
        if j not in (route_j[0], route_j[-1]):
            continue
        
        # Determine the order of merging
        # Case 1: i is end of route_i, j is start of route_j -> merge route_i + route_j
        # Case 2: i is start of route_i, j is end of route_j -> merge route_j + route_i
        # Case 3: i is end of route_i, j is end of route_j -> merge route_i + reversed(route_j)
        # Case 4: i is start of route_i, j is start of route_j -> merge reversed(route_i) + route_j
        
        new_route = None
        if i == route_i[-1] and j == route_j[0]:
            new_route = route_i + route_j
        elif i == route_i[0] and j == route_j[-1]:
            new_route = route_j + route_i
        elif i == route_i[-1] and j == route_j[-1]:
            new_route = route_i + route_j[::-1]
        elif i == route_i[0] and j == route_j[0]:
            new_route = route_i[::-1] + route_j
        
        if new_route is None:
            continue
            
        # Check feasibility of the new merged route
        feasible, _ = _check_route(new_route, instance, D)
        if feasible:
            # Merge routes
            # Remove route_j from list (assuming ri < rj for simplicity, or handle indices carefully)
            if ri < rj:
                routes[ri] = new_route
                del routes[rj]
                # Update route_of_customer for customers in route_j
                for c in route_j:
                    route_of_customer[c] = ri
            else:
                routes[rj] = new_route
                del routes[ri]
                for c in route_i:
                    route_of_customer[c] = rj
            # Update route_of_customer for customers in the remaining part of the deleted route? No, they are now in the new route.
            # We need to update the index for the surviving route if we deleted the lower index.
            # The logic above handles the deletion. We just need to ensure route_of_customer is updated for all nodes in the new route.
            for c in new_route:
                # Find the current index of the route containing c
                # Since we just merged, we know which index it is now
                pass # The updates in the if/else blocks cover this partially, let's be explicit.
            
            # Re-evaluate indices for safety
            # This is O(N^2) but N is small (25)
            for idx, r in enumerate(routes):
                for c in r:
                    route_of_customer[c] = idx

    # Filter out empty routes (shouldn't happen)
    routes = [r for r in routes if r]
    return routes

def _local_search_improved(routes, instance, D, deadline):
    """Perform 2-opt and relocate local search with time limit."""
    n_customers = instance["n_customers"]
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 1. 2-Opt within each route
        for i in range(len(routes)):
            route = routes[i]
            if len(route) < 2:
                continue
            for a in range(len(route)):
                for b in range(a + 1, len(route)):
                    # Reverse segment a..b
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    feasible, _ = _check_route(new_route, instance, D)
                    if feasible:
                        old_dist = _route_distance(route, instance, D)
                        new_dist = _route_distance(new_route, instance, D)
                        if new_dist < old_dist - 1e-9:
                            routes[i] = new_route
                            improved = True
        
        # 2. Relocate customer between routes
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                customer = routes[i][j]
                # Remove customer
                new_route_i = routes[i][:j] + routes[i][j+1:]
                if not new_route_i:
                    # Route becomes empty
                    feasible_i = True
                else:
                    feasible_i, _ = _check_route(new_route_i, instance, D)
                
                if not feasible_i:
                    continue
                
                # Try inserting into other routes
                for k in range(len(routes)):
                    if k == i:
                        continue
                    pos, feasible, _ = _try_insert(customer, routes[k], instance, D)
                    if feasible:
                        # Check if this improves total distance
                        old_dist_i = _route_distance(routes[i], instance, D)
                        old_dist_k = _route_distance(routes[k], instance, D)
                        
                        new_dist_i = _route_distance(new_route_i, instance, D) if new_route_i else 0.0
                        new_route_k = routes[k][:pos] + [customer] + routes[k][pos:]
                        new_dist_k = _route_distance(new_route_k, instance, D)
                        
                        if new_dist_i + new_dist_k < old_dist_i + old_dist_k - 1e-9:
                            routes[i] = new_route_i
                            routes[k] = new_route_k
                            improved = True
                            # If route i became empty, remove it
                            if not routes[i]:
                                del routes[i]
                                # Adjust indices for k if k > i
                                if k > i:
                                    k -= 1
                                # Break out of loops to restart search as indices changed
                                break
                if improved:
                    break
            if improved:
                break
                
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115  # Leave margin for safety
    
    # Build distance matrix
    D = _build_dist_matrix(instance)
    n_customers = instance["n_customers"]
    
    # 1. Construct initial routes using Savings Heuristic
    routes = _savings_construction(instance, D)
    
    # 2. Local Search
    routes = _local_search_improved(routes, instance, D, deadline)
    
    # 3. Verify all customers are served
    served = set()
    for route in routes:
        for v in route:
            if v <= n_customers:
                served.add(v)
    
    # Add any missing customers as single routes (should not happen with savings, but safety)
    for c in range(1, n_customers + 1):
        if c not in served:
            routes.append([c])
    
    # Final verification
    valid_routes = []
    for route in routes:
        feasible, dist = _check_route(route, instance, D)
        if feasible:
            valid_routes.append(route)
    
    return valid_routes
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
