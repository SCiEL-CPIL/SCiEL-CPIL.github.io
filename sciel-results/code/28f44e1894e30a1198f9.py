<<<<<<< SEARCH
def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    best_routes = None
    best_dist = float('inf')

    # Phase 1: Generate diverse initial solutions
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 35:
            break
        routes = _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Cluster-based constructions
    for i in range(5):
        if time.time() > deadline - 30:
            break
        routes = _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity,
                                     n_clusters=4 + i, seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Dijkstra-based route construction for small clusters
    for i in range(5):
        if time.time() > deadline - 25:
            break
        routes = _dijkstra_route_build(instance, dist_matrix, n_customers, stations_set,
                                        battery_max, speed, recharge_rate, horizon, capacity,
                                        seed=i, max_customers=6)
        if not routes:
            continue
        # Need to cover all customers; this is just one route, so skip for now
        # Actually this returns a single route, not full solution. Skip.
        pass

    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local search
    if time.time() < deadline - 20:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 20)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 10)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Polish
    if time.time() < deadline - 5:
        polished = _polish_routes(best_routes, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(polished, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = polished

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
=======
def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    ctx = _build_ctx(instance)
    n_customers = ctx["n_customers"]

    best_routes = None
    best_dist = float('inf')

    # Phase 1: Generate diverse initial solutions
    n_starts = 12
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        routes = _greedy_construct(instance, dist_matrix, ctx, seed=i)
        valid, dist = _validate_solution(routes, instance, dist_matrix, ctx)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Cluster-based constructions
    for i in range(6):
        if time.time() > deadline - 35:
            break
        routes = _cluster_construct(instance, dist_matrix, ctx,
                                     n_clusters=4 + i, seed=i)
        valid, dist = _validate_solution(routes, instance, dist_matrix, ctx)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local search
    if time.time() < deadline - 25:
        improved = _local_search(best_routes, dist_matrix, ctx, deadline - 25)
        valid, dist = _validate_solution(improved, instance, dist_matrix, ctx)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, ctx, deadline - 8)
        valid, dist = _validate_solution(sa_routes, instance, dist_matrix, ctx)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Polish
    if time.time() < deadline - 3:
        polished = _polish_routes(best_routes, dist_matrix, ctx, deadline - 1)
        valid, dist = _validate_solution(polished, instance, dist_matrix, ctx)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = polished

    # Final validation
    valid, dist = _validate_solution(best_routes, instance, dist_matrix, ctx)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
>>>>>>> REPLACE
