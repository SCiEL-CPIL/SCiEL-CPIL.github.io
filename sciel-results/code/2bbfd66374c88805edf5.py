<<<<<<< SEARCH
def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    # Try multiple construction heuristics and keep the best
    best_routes = None
    best_dist = float('inf')

    # 1. Nearest neighbor / greedy insertion
    if time.time() < deadline - 30:
        routes = _nearest_neighbor_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = routes
            best_dist = dist

    # 2. Savings heuristic
    if time.time() < deadline - 30:
        routes = _savings_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = routes
            best_dist = dist

    # 3. Cluster-based construction
    if time.time() < deadline - 30:
        routes = _cluster_based_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = routes
            best_dist = dist

    if best_routes is None:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Optimize each route with 2-opt and OR-opt
    for r_idx in range(len(best_routes)):
        if time.time() > deadline - 10:
            break
        best_routes[r_idx] = _two_opt_route(
            best_routes[r_idx], dist_matrix, instance["n_customers"],
            set(instance["stations"]), instance["battery"], instance["speed"],
            instance["recharge_rate"], instance["horizon"],
            np.array(instance["demands"], dtype=np.float64),
            np.array(instance["ready"], dtype=np.float64),
            np.array(instance["due"], dtype=np.float64),
            np.array(instance["service"], dtype=np.float64),
            max_iters=25
        )
        if time.time() > deadline - 10:
            break
        best_routes[r_idx] = _or_opt_route(
            best_routes[r_idx], dist_matrix, instance["n_customers"],
            set(instance["stations"]), instance["battery"], instance["speed"],
            instance["recharge_rate"], instance["horizon"],
            np.array(instance["demands"], dtype=np.float64),
            np.array(instance["ready"], dtype=np.float64),
            np.array(instance["due"], dtype=np.float64),
            np.array(instance["service"], dtype=np.float64),
            max_iters=15
        )
    
    # Final high-precision polish pass with tighter tolerance and more iterations
    if time.time() < deadline - 5:
        for r_idx in range(len(best_routes)):
            if time.time() > deadline - 3:
                break
            # More aggressive 2-opt with higher iteration count
            polished = _two_opt_route(
                best_routes[r_idx], dist_matrix, instance["n_customers"],
                set(instance["stations"]), instance["battery"], instance["speed"],
                instance["recharge_rate"], instance["horizon"],
                np.array(instance["demands"], dtype=np.float64),
                np.array(instance["ready"], dtype=np.float64),
                np.array(instance["due"], dtype=np.float64),
                np.array(instance["service"], dtype=np.float64),
                max_iters=40
            )
            if _compute_total_distance([polished], dist_matrix) < _compute_total_distance([best_routes[r_idx]], dist_matrix):
                best_routes[r_idx] = polished

    # Inter-route search
    if time.time() < deadline - 15:
        new_routes = _inter_route_search(best_routes, dist_matrix, instance, deadline - 5)
        new_dist = _compute_total_distance(new_routes, dist_matrix)
        if new_dist < best_dist:
            best_routes = new_routes
            best_dist = new_dist

    # Simulated annealing
    if time.time() < deadline - 10:
        new_routes = _simulated_annealing(best_routes, dist_matrix, instance, deadline - 2)
        new_dist = _compute_total_distance(new_routes, dist_matrix)
        if new_dist < best_dist:
            best_routes = new_routes
            best_dist = new_dist
    
    # Final station optimization pass
    if time.time() < deadline - 2:
        for r_idx in range(len(best_routes)):
            if time.time() > deadline - 1:
                break
            optimized = _optimize_stations_in_route(
                best_routes[r_idx], dist_matrix, instance["n_customers"],
                set(instance["stations"]), instance["battery"], instance["speed"],
                instance["recharge_rate"], instance["horizon"],
                np.array(instance["demands"], dtype=np.float64),
                np.array(instance["ready"], dtype=np.float64),
                np.array(instance["due"], dtype=np.float64),
                np.array(instance["service"], dtype=np.float64)
            )
            if _compute_total_distance([optimized], dist_matrix) < _compute_total_distance([best_routes[r_idx]], dist_matrix):
                best_routes[r_idx] = optimized

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]

    return best_routes
=======
def _greedy_fallback_construct(instance, dist_matrix):
    """Simple greedy fallback construction."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    
    routes = []
    route_loads = []
    
    for cust in customers:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue
                if pos == 0:
                    cost = (dist_matrix[0, cust] + dist_matrix[cust, route[0]] -
                           dist_matrix[0, route[0]])
                elif pos == len(route):
                    cost = (dist_matrix[route[-1], cust] + dist_matrix[cust, 0] -
                           dist_matrix[route[-1], 0])
                else:
                    cost = (dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] -
                           dist_matrix[route[pos-1], route[pos]])
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = r_idx
                    best_insert_pos = pos
        
        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_insert_pos, cust)
            route_loads[best_route_idx] += demands[cust]
    
    return routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    # Primary: Dijkstra-based construction
    best_routes = None
    best_dist = float('inf')

    if time.time() < deadline - 40:
        routes = _dijkstra_construct(instance)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = routes
            best_dist = dist

    # Fallback if Dijkstra failed
    if best_routes is None and time.time() < deadline - 30:
        routes = _greedy_fallback_construct(instance, dist_matrix)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_routes = routes
            best_dist = dist

    if best_routes is None:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Optimize each route with 2-opt and OR-opt
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations_set = set(instance["stations"])
    
    for r_idx in range(len(best_routes)):
        if time.time() > deadline - 15:
            break
        best_routes[r_idx] = _two_opt_route(
            best_routes[r_idx], dist_matrix, instance["n_customers"],
            stations_set, instance["battery"], instance["speed"],
            instance["recharge_rate"], instance["horizon"],
            demands, ready, due, service,
            max_iters=20
        )
        if time.time() > deadline - 15:
            break
        best_routes[r_idx] = _or_opt_route(
            best_routes[r_idx], dist_matrix, instance["n_customers"],
            stations_set, instance["battery"], instance["speed"],
            instance["recharge_rate"], instance["horizon"],
            demands, ready, due, service,
            max_iters=10
        )

    # Inter-route search
    if time.time() < deadline - 20:
        new_routes = _inter_route_search(best_routes, dist_matrix, instance, deadline - 5)
        new_dist = _compute_total_distance(new_routes, dist_matrix)
        if new_dist < best_dist:
            best_routes = new_routes
            best_dist = new_dist

    # Simulated annealing
    if time.time() < deadline - 10:
        new_routes = _simulated_annealing(best_routes, dist_matrix, instance, deadline - 3)
        new_dist = _compute_total_distance(new_routes, dist_matrix)
        if new_dist < best_dist:
            best_routes = new_routes
            best_dist = new_dist

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]

    return best_routes
>>>>>>> REPLACE
