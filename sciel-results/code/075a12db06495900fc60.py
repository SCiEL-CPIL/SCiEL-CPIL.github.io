"""EV routing with time windows and battery constraints - improved search-based approach."""
import numpy as np
import time
import random
import math

DEADLINE_SECONDS = 145.0  # Safety margin below 150s limit

def _build_dist_matrix(coords):
    coords = np.array(coords, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))

def _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """
    Simulates a route to check feasibility and calculate total distance.
    Returns (valid, total_distance, final_time, final_battery)
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations_set:
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time_at += energy_needed * recharge_rate
            battery = battery_max

        current_node = node

    # Return to depot
    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d
    distance += d

    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0

    return True, distance, time_at, battery

def _validate_solution(routes, instance):
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    dist_matrix = _build_dist_matrix(coords)
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    served = set()
    total_distance = 0.0

    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                             stations_set, battery_max, speed,
                                             recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_distance += dist
        for node in route:
            if 1 <= node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance

def _compute_total_distance(routes, dist_matrix):
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total

def _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed,
                                      recharge_rate, horizon, capacity)
    return valid

def _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """
    Greedy construction: Sort customers by due date (or random), 
    insert each into the best position in the best existing route.
    """
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    # Strategy: Sort by due date, but add some randomness to diversify
    customers = list(range(1, n_customers + 1))
    # 50% chance to sort by due, 50% random
    if random.random() < 0.5:
        customers.sort(key=lambda x: due[x])
    else:
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

def _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """
    Cluster-based construction:
    1. K-Means clustering of customers into K clusters.
    2. For each cluster, build a route using Nearest Neighbor heuristic.
    3. Check feasibility and repair if necessary (by splitting or moving to other routes).
    """
    random.seed(seed)
    demands = instance["demands"]
    coords = np.array(instance["coords"], dtype=np.float64)
    
    # Estimate number of clusters based on capacity
    total_demand = sum(demands[1:n_customers+1])
    est_k = max(2, int(np.ceil(total_demand / capacity)) + 1) # Add 1 for safety
    est_k = min(est_k, 10) # Cap at 10 for small instances
    
    # Simple K-Means implementation
    # Initialize centroids randomly
    customer_ids = list(range(1, n_customers + 1))
    customer_coords = coords[1:n_customers+1]
    
    # Random init
    centroids = customer_coords[random.sample(range(n_customers), est_k)]
    
    for _ in range(10): # 10 iterations
        # Assign to nearest centroid
        clusters = [[] for _ in range(est_k)]
        for i, c_id in enumerate(customer_ids):
            dists = np.linalg.norm(centroids - customer_coords[i], axis=1)
            closest = np.argmin(dists)
            clusters[closest].append(c_id)
        
        # Update centroids
        new_centroids = []
        for k in range(est_k):
            if clusters[k]:
                new_centroids.append(np.mean(customer_coords[[c_id-1 for c_id in clusters[k]]], axis=0))
            else:
                new_centroids.append(centroids[k])
        centroids = np.array(new_centroids)
        
        # Check convergence
        if all(np.allclose(c1, c2) for c1, c2 in zip(centroids, new_centroids)):
            break
    
    # Build routes for each cluster
    routes = []
    for k in range(est_k):
        if not clusters[k]:
            continue
        route = []
        unvisited = set(clusters[k])
        current_node = 0
        current_battery = battery_max
        current_time = 0.0
        current_load = 0.0
        
        while unvisited:
            # Find nearest feasible customer
            best_cust = -1
            best_dist = float('inf')
            
            for c_id in unvisited:
                d = dist_matrix[current_node, c_id]
                # Check battery
                if current_battery - d < 0:
                    # Try to go to station? For simplicity, skip if infeasible directly
                    # In a real implementation, we'd check if we can reach a station and then customer
                    # But for construction, we assume clusters are compact enough
                    continue
                # Check time window (roughly)
                arrival_time = current_time + d / speed
                if arrival_time > instance["due"][c_id]:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_cust = c_id
            
            if best_cust == -1:
                # No feasible customer from current node. 
                # Try to go to depot and restart? Or just pick nearest regardless (will be infeasible)
                # For now, pick nearest unvisited to avoid infinite loop, will fail validation
                best_cust = min(unvisited, key=lambda c: dist_matrix[current_node, c])
                best_dist = dist_matrix[current_node, best_cust]
                
            # Move to best_cust
            d = dist_matrix[current_node, best_cust]
            current_time += d / speed
            current_battery -= d
            current_load += demands[best_cust]
            
            if current_time < instance["ready"][best_cust]:
                current_time = instance["ready"][best_cust]
            current_time += instance["service"][best_cust]
            
            route.append(best_cust)
            unvisited.remove(best_cust)
            current_node = best_cust
            
            # Check if we need to return to depot (battery or time)
            # Simple heuristic: if battery is low or time is high, close route
            d_to_depot = dist_matrix[current_node, 0]
            if current_battery - d_to_depot < 0:
                # Can't return to depot. This route is infeasible.
                # We will handle this by splitting or moving customers later.
                # For now, break and mark as invalid
                break
            if current_time + d_to_depot / speed > horizon:
                break
            
            # Heuristic: if load is high, close route
            if current_load > capacity * 0.8:
                break
                
        # If we broke out early, we have an infeasible route. 
        # We will just add what we have and let validation/local search fix it? 
        # No, better to ensure feasibility. 
        # If the route is infeasible, we discard it and its customers will be handled by greedy fallback?
        # Let's just return the routes as is. The local search will fix it.
        if route:
            routes.append(route)
            
    # Fallback for any customers not assigned (should be none if clustering worked)
    assigned = set()
    for r in routes:
        for c in r:
            assigned.add(c)
            
    missing = [c for c in range(1, n_customers+1) if c not in assigned]
    if missing:
        # Use greedy for missing
        routes_missing = _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                                           battery_max, speed, recharge_rate, horizon, capacity, seed=seed+100)
        # Merge? This is complex. Just append.
        routes.extend(routes_missing)
        
    return routes

def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=50):
    """2-opt improvement within a route."""
    if len(route) < 2:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if is_feasible(new_route):
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break

    return best_route

def _or_opt(route, dist_matrix, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity, max_iters=50):
    """OR-opt: Move segments of length 1, 2, 3 within the route."""
    if len(route) < 3:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        n = len(best_route)
        for seg_len in [1, 2, 3]:
            if n < seg_len:
                continue
            for i in range(n - seg_len + 1):
                segment = best_route[i:i+seg_len]
                rest = best_route[:i] + best_route[i+seg_len:]
                # Try inserting segment at all positions in 'rest'
                for j in range(len(rest) + 1):
                    if i > 0 and j >= i:
                        # Avoid invalid moves where segment is effectively staying in place or overlapping logic
                        # This is a heuristic check; strict validation is done by feasibility
                        pass 
                    new_route = rest[:j] + segment + rest[j:]
                    # Skip if it's the same route
                    if new_route == best_route:
                        continue
                    if is_feasible(new_route):
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break

    return best_route

def _local_search(routes, dist_matrix, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Local search with 2-opt, OR-opt, and inter-route moves."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)
    start_time = time.time()

    # Intra-route improvements
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        # 2-opt
        routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, max_iters=20)
        # OR-opt
        routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                 stations_set, battery_max, speed, recharge_rate,
                                 horizon, capacity, max_iters=20)
    
    current_dist = get_total_dist(routes)

    # Inter-route moves: Relocate and Swap
    max_passes = 30
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False

        # Relocate
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Swap
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust1 = routes[r1][pos1]
                if not (1 <= cust1 <= n_customers):
                    continue
                for r2 in range(r1 + 1, len(routes)):
                    if time.time() > deadline:
                        break
                    for pos2 in range(len(routes[r2])):
                        if time.time() > deadline:
                            break
                        cust2 = routes[r2][pos2]
                        if not (1 <= cust2 <= n_customers):
                            continue
                        
                        # Check capacity
                        load1 = route_load(routes[r1])
                        load2 = route_load(routes[r2])
                        if load1 - demands[cust1] + demands[cust2] > capacity:
                            continue
                        if load2 - demands[cust2] + demands[cust1] > capacity:
                            continue
                        
                        route1_new = routes[r1][:]
                        route2_new = routes[r2][:]
                        route1_new[pos1] = cust2
                        route2_new[pos2] = cust1
                        
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break

    routes = [r for r in routes if len(r) > 0]
    return routes

def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Simulated annealing to escape local optima."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 10.0
    end_temp = 0.05
    max_iters = 5000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.4:
            # Move a customer
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
        elif perturbation_type < 0.7:
            # Swap two customers
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        else:
            # Reverse segment
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]

        new_routes = [r for r in new_routes if len(r) > 0]
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist

        temp *= cooling_rate

    return best_routes

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
    # Use both Greedy and Cluster-based construction
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline - 40:
            break
        # Alternate between Greedy and Cluster
        if i % 2 == 0:
            routes = _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                                        battery_max, speed, recharge_rate, horizon, capacity,
                                        seed=i)
        else:
            routes = _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                                        battery_max, speed, recharge_rate, horizon, capacity,
                                        seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    if best_routes is None:
        # Fallback
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local search on best solution
    if time.time() < deadline - 20:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 20)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated annealing
    if time.time() < deadline - 10:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 5)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
