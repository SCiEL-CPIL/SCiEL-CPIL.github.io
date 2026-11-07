<<<<<<< SEARCH
def _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """Greedy nearest-neighbor with time window awareness."""
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)

    routes = []
    route_loads = []

    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                if _route_feasible(new_route, dist_matrix, instance, n_customers,
                                    stations_set, battery_max, speed, recharge_rate,
                                    horizon, capacity):
                    # Approximate cost: distance added
                    if pos == 0:
                        cost = dist_matrix[0, cust] + dist_matrix[cust, route[0]] - dist_matrix[0, route[0]]
                    elif pos == len(route):
                        cost = dist_matrix[route[-1], cust] + dist_matrix[cust, 0] - dist_matrix[route[-1], 0]
                    else:
                        cost = dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] - dist_matrix[route[pos-1], route[pos]]
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos

        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_pos, cust)
            route_loads[best_route_idx] += demands[cust]

    return routes
=======
def _savings_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """Clarke-Wright Savings heuristic adapted for EV constraints."""
    demands = instance["demands"]
    
    # Initialize each customer as its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]
    
    # Calculate savings for merging routes i and j
    # Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            savings.append((s, i, j))
    
    # Sort savings in descending order
    savings.sort(reverse=True)
    
    # Map customer to route index
    cust_to_route = {i: i - 1 for i in range(1, n_customers + 1)}
    
    for s, i, j in savings:
        if s <= 0:
            break
            
        ri = cust_to_route[i]
        rj = cust_to_route[j]
        
        if ri == rj:
            continue
            
        # Check capacity
        if route_loads[ri] + route_loads[rj] > capacity:
            continue
            
        # Determine the tail of ri and head of rj
        route_i = routes[ri]
        route_j = routes[rj]
        
        # We can merge in two ways:
        # 1. Append route_j to route_i (if route_i ends at i and route_j starts at j)
        # 2. Prepend route_j to route_i (if route_i starts at i and route_j ends at j)
        # 3. Append route_i to route_j (if route_j ends at j and route_i starts at i)
        # 4. Prepend route_i to route_j (if route_j starts at j and route_i ends at i)
        
        # Actually, standard savings merges the routes by connecting i and j.
        # We need to check if i is at the end of route_i and j is at the start of route_j, etc.
        
        feasible = False
        new_route = None
        
        # Case 1: route_i ends with i, route_j starts with j
        if route_i[-1] == i and route_j[0] == j:
            candidate = route_i + route_j
            if _route_feasible(candidate, dist_matrix, instance, n_customers,
                               stations_set, battery_max, speed, recharge_rate,
                               horizon, capacity):
                feasible = True
                new_route = candidate
        # Case 2: route_i starts with i, route_j ends with j
        elif route_i[0] == i and route_j[-1] == j:
            candidate = route_j + route_i
            if _route_feasible(candidate, dist_matrix, instance, n_customers,
                               stations_set, battery_max, speed, recharge_rate,
                               horizon, capacity):
                feasible = True
                new_route = candidate
        # Case 3: route_i ends with i, route_j ends with j -> reverse route_j? No, we don't reverse arbitrarily in savings usually, but we can try.
        # Actually, the standard savings algorithm assumes routes are directed.
        # We will only merge if the specific endpoints match.
        
        if feasible:
            # Merge
            routes[ri] = new_route
            route_loads[ri] += route_loads[rj]
            routes[rj] = []
            route_loads[rj] = 0
            # Update cust_to_route for all customers in route_j
            for cust in new_route:
                if cust <= n_customers:
                    cust_to_route[cust] = ri
                    
    # Remove empty routes
    routes = [r for r in routes if len(r) > 0]
    return routes
>>>>>>> REPLACE

Now, let's add Or-Opt to the local search. Or-Opt moves a segment of size 1, 2, or 3 from one position to another.

