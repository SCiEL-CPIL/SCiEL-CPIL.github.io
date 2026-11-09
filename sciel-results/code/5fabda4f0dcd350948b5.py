"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt
from collections import defaultdict

def _get_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    n = len(coords)
    arr = np.array(coords)
    dx = arr[:, 0, None] - arr[None, :, 0]
    dy = arr[:, 1, None] - arr[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    return dist

def _route_distance_fast(dist_matrix, route):
    """Calculate total distance of a route using precomputed distance matrix."""
    if not route:
        return 0.0
    d = dist_matrix[0][route[0]]
    for k in range(len(route) - 1):
        d += dist_matrix[route[k]][route[k + 1]]
    d += dist_matrix[route[-1]][0]
    return d

def _check_route_feasibility_fast(dist_matrix, demands, ready, due, service,
                              battery, recharge_rate, horizon, route,
                              capacity, n_customers, stations):
    """
    Check if a single route is feasible using precomputed distances.
    Returns (feasible, total_distance).
    """
    if not route:
        return True, 0.0

    # Check capacity
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
            if total_demand > capacity:
                return False, 0.0

    # Simulate
    t = 0.0
    b = battery
    prev = 0
    total_dist = 0.0
    
    for v in route:
        d = dist_matrix[prev, v]
        t += d
        b -= d
        total_dist += d
        if b < -1e-9:
            return False, 0.0
        
        if v <= n_customers:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
        else:
            # Station
            time_charging = (battery - b) * recharge_rate
            t += time_charging
            b = battery  # Refill to full
        
        prev = v

    # Return to depot
    d = dist_matrix[prev, 0]
    t += d
    b -= d
    total_dist += d
    if b < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist

def _dp_route_search(customers, dist_matrix, demands, ready, due, service,
                     battery, recharge_rate, horizon, capacity, n_customers, stations,
                     deadline):
    """
    Multi-objective DP over (customer, time_quantized, battery_quantized) states.
    Finds minimum-distance feasible route for the given set of customers.
    Returns (route_list, total_distance) or (None, inf) if no feasible route found.
    """
    if not customers:
        return [], 0.0
    
    n_c = len(customers)
    if n_c > 18:
        return None, float('inf')  # Too many customers for bitmask DP
    
    # Create bitmask for customers
    cust_to_idx = {c: i for i, c in enumerate(customers)}
    all_mask = (1 << n_c) - 1
    
    # Time and battery quantization
    time_step = 2.0  # 2-second granularity
    batt_step = 5.0  # 5-unit battery granularity
    
    def quantize_time(t):
        return int(round(t / time_step))
    
    def quantize_batt(b):
        return int(round(b / batt_step))
    
    # DP state: (mask, last_customer_idx, time_quant, batt_quant) -> (min_dist, path)
    # Use dict for sparse storage
    # State: (mask, last_idx, tq, bq) -> (dist, prev_state)
    dp = {}
    
    # Initial state: at depot (node 0), no customers visited
    # We'll track the "last node" which can be 0 (depot) or a customer index
    # For depot, we use last_idx = -1
    
    # Initialize: try going to each customer from depot
    for i, c in enumerate(customers):
        d = dist_matrix[0, c]
        if d > battery + 1e-9:
            continue
        t_arrive = d
        b_arrive = battery - d
        
        if b_arrive < -1e-9:
            continue
        
        # Check time window
        if t_arrive < ready[c]:
            t_arrive = ready[c]
        if t_arrive > due[c] + 1e-9:
            continue
        t_after_service = t_arrive + service[c]
        
        tq = quantize_time(t_after_service)
        bq = quantize_batt(b_arrive)
        
        mask = 1 << i
        state = (mask, i, tq, bq)
        if state not in dp or d < dp[state][0]:
            dp[state] = (d, None)  # (dist, prev_state)
    
    # Also consider going to a station first
    for s in stations:
        d = dist_matrix[0, s]
        if d > battery + 1e-9:
            continue
        t_arrive = d
        b_arrive = battery - d
        if b_arrive < -1e-9:
            continue
        # At station: recharge
        charge_time = (battery - b_arrive) * recharge_rate
        t_after = t_arrive + charge_time
        b_after = battery  # Full battery
        
        # From station, go to each customer
        for i, c in enumerate(customers):
            d2 = dist_matrix[s, c]
            if d2 > battery + 1e-9:
                continue
            t2 = t_after + d2
            b2 = battery - d2
            if b2 < -1e-9:
                continue
            if t2 < ready[c]:
                t2 = ready[c]
            if t2 > due[c] + 1e-9:
                continue
            t2 += service[c]
            
            tq = quantize_time(t2)
            bq = quantize_batt(b2)
            
            mask = 1 << i
            state = (mask, i, tq, bq)
            total_d = d + d2
            if state not in dp or total_d < dp[state][0]:
                dp[state] = (total_d, ('station', s, 0, quantize_time(0), quantize_batt(battery)))
    
    # DP expansion
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Get all states to expand
        states_to_expand = list(dp.keys())
        
        for state in states_to_expand:
            mask, last_idx, tq, bq = state
            dist_so_far, _ = dp[state]
            
            if mask == all_mask:
                continue  # All customers visited
            
            last_node = customers[last_idx]
            t_est = tq * time_step
            b_est = bq * batt_step
            
            # Try visiting each unvisited customer
            for i, c in enumerate(customers):
                if mask & (1 << i):
                    continue
                
                d = dist_matrix[last_node, c]
                if d > b_est + 1e-9:
                    # Need to go through a station
                    best_via_station = None
                    best_via_dist = float('inf')
                    
                    for s in stations:
                        d1 = dist_matrix[last_node, s]
                        if d1 > b_est + 1e-9:
                            continue
                        b_mid = b_est - d1
                        if b_mid < -1e-9:
                            continue
                        t_mid = t_est + d1
                        charge_time = (battery - b_mid) * recharge_rate
                        t_mid += charge_time
                        b_mid = battery
                        
                        d2 = dist_matrix[s, c]
                        if d2 > battery + 1e-9:
                            continue
                        t2 = t_mid + d2
                        b2 = battery - d2
                        if b2 < -1e-9:
                            continue
                        
                        if t2 < ready[c]:
                            t2 = ready[c]
                        if t2 > due[c] + 1e-9:
                            continue
                        t2 += service[c]
                        
                        total_d = d1 + d2
                        if total_d < best_via_dist:
                            best_via_dist = total_d
                            best_via_station = (s, t2, b2)
                    
                    if best_via_station is None:
                        continue
                    s, t2, b2 = best_via_station
                    new_tq = quantize_time(t2)
                    new_bq = quantize_batt(b2)
                    new_mask = mask | (1 << i)
                    new_state = (new_mask, i, new_tq, new_bq)
                    new_dist = dist_so_far + best_via_dist
                    if new_state not in dp or new_dist < dp[new_state][0] - 1e-9:
                        dp[new_state] = (new_dist, state)
                        improved = True
                    continue
                
                # Direct visit
                t_new = t_est + d
                b_new = b_est - d
                if b_new < -1e-9:
                    continue
                
                if t_new < ready[c]:
                    t_new = ready[c]
                if t_new > due[c] + 1e-9:
                    continue
                t_new += service[c]
                
                new_tq = quantize_time(t_new)
                new_bq = quantize_batt(b_new)
                new_mask = mask | (1 << i)
                new_state = (new_mask, i, new_tq, new_bq)
                new_dist = dist_so_far + d
                
                if new_state not in dp or new_dist < dp[new_state][0] - 1e-9:
                    dp[new_state] = (new_dist, state)
                    improved = True
    
    # Find best complete route
    best_dist = float('inf')
    best_state = None
    for state, (dist, _) in dp.items():
        mask, last_idx, tq, bq = state
        if mask != all_mask:
            continue
        
        last_node = customers[last_idx]
        d_return = dist_matrix[last_node, 0]
        if d_return > bq * batt_step + 1e-9:
            # Check if can return via station
            can_return = False
            for s in stations:
                d1 = dist_matrix[last_node, s]
                if d1 > bq * batt_step + 1e-9:
                    continue
                # Can reach station, then return to depot
                d2 = dist_matrix[s, 0]
                if d2 > battery + 1e-9:
                    continue
                can_return = True
                break
            if not can_return:
                continue
        
        total_dist = dist + d_return
        if total_dist < best_dist:
            best_dist = total_dist
            best_state = state
    
    if best_state is None:
        return None, float('inf')
    
    # Reconstruct route
    route = []
    state = best_state
    while state is not None:
        mask, last_idx, tq, bq = state
        route.append(customers[last_idx])
        _, prev_state = dp[state]
        state = prev_state
    
    route.reverse()
    
    # Verify the route
    feasible, actual_dist = _check_route_feasibility_fast(
        dist_matrix, demands, ready, due, service,
        battery, recharge_rate, horizon, route,
        capacity, n_customers, stations
    )
    
    if feasible:
        return route, actual_dist
    else:
        return None, float('inf')

def _two_opt_route(route, dist_matrix, demands, ready, due, service,
                   battery, recharge_rate, horizon, capacity, n_customers, stations,
                   deadline):
    """Apply 2-opt moves within a single route to reduce distance while maintaining feasibility."""
    if len(route) < 3:
        return route, 0.0
    
    n = len(route)
    improved = True
    total_improvement = 0.0
    
    while improved:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                return route, total_improvement
            for j in range(i + 1, n):
                # Compute distance delta for reversing segment [i..j]
                # Before: ... a[i-1] - a[i] ... a[j] - a[j+1] ...
                # After:  ... a[i-1] - a[j] ... a[i] - a[j+1] ...
                # where a[-1] = depot(0), a[n] = depot(0)
                
                a = 0 if i == 0 else route[i - 1]
                b_node = route[i]
                c = route[j]
                d_node = 0 if j == n - 1 else route[j + 1]
                
                old_d = dist_matrix[a, b_node] + dist_matrix[c, d_node]
                new_d = dist_matrix[a, c] + dist_matrix[b_node, d_node]
                
                if new_d >= old_d - 1e-9:
                    continue
                
                # Try the reversal
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                # Check feasibility
                feasible, _ = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                
                if feasible:
                    route = new_route
                    total_improvement += old_d - new_d
                    improved = True
                    break
    
    return route, total_improvement

def _or_opt_route(route, dist_matrix, demands, ready, due, service,
                  battery, recharge_rate, horizon, capacity, n_customers, stations,
                  deadline):
    """Apply Or-opt moves (relocate 1-3 consecutive customers) to reduce distance."""
    if len(route) < 3:
        return route, 0.0
    
    n = len(route)
    total_improvement = 0.0
    improved = True
    
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() > deadline:
                return route, total_improvement
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    return route, total_improvement
                segment = route[i:i+seg_len]
                
                # Remove segment
                remaining = route[:i] + route[i+seg_len:]
                if not remaining:
                    continue
                
                # Try inserting segment at each position in remaining
                for j in range(len(remaining) + 1):
                    if j == i or j == i + seg_len:
                        continue  # Skip original position
                    
                    new_route = remaining[:j] + segment + remaining[j:]
                    
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    
                    if feasible:
                        old_dist = _route_distance_fast(dist_matrix, route)
                        if nd < old_dist - 1e-9:
                            route = new_route
                            total_improvement += old_dist - nd
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return route, total_improvement

def _cluster_customers(coords, n_customers, n_clusters):
    """K-means clustering of customers (1..n_customers) into n_clusters groups."""
    import random
    
    customer_coords = np.array(coords[1:n_customers+1])
    n = len(customer_coords)
    
    if n <= n_clusters:
        return [[i+1] for i in range(n)]
    
    # Initialize centroids randomly
    indices = np.random.choice(n, n_clusters, replace=False)
    centroids = customer_coords[indices].copy()
    
    for _ in range(50):
        # Assign points to nearest centroid
        dists = np.sqrt(((customer_coords[:, None, :] - centroids[None, :, :])**2).sum(axis=2))
        labels = np.argmin(dists, axis=1)
        
        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centroids[k] = customer_coords[mask].mean(axis=0)
            else:
                new_centroids[k] = centroids[k]
        
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids
    
    # Build clusters
    clusters = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        clusters[label].append(i + 1)
    
    return [c for c in clusters if c]

def _nearest_neighbor_route(customers, dist_matrix, demands, ready, due, service,
                           battery, recharge_rate, horizon, capacity, n_customers, stations,
                           deadline):
    """Build a route using nearest neighbor heuristic for given customers."""
    if not customers:
        return []
    
    route = []
    unvisited = set(customers)
    current = 0  # Start at depot
    t = 0.0
    b = battery
    total_demand = 0
    
    while unvisited and time.time() < deadline:
        best_c = None
        best_dist = float('inf')
        
        for c in unvisited:
            if total_demand + demands[c] > capacity:
                continue
            
            d = dist_matrix[current, c]
            new_t = t + d
            new_b = b - d
            
            if new_b < -1e-9:
                continue
            
            if new_t < ready[c]:
                new_t = ready[c]
            if new_t > due[c] + 1e-9:
                continue
            new_t += service[c]
            
            if d < best_dist:
                best_dist = d
                best_c = c
        
        if best_c is None:
            # Try with a station visit first if battery is low
            if b < 100:
                best_station = None
                best_s_dist = float('inf')
                for s in stations:
                    d = dist_matrix[current, s]
                    if d < best_s_dist:
                        best_s_dist = d
                        best_station = s
                
                if best_station is not None:
                    t += best_s_dist
                    b -= best_s_dist
                    t += (battery - b) * recharge_rate
                    b = battery
                    current = best_station
                    continue
            
            break
        
        route.append(best_c)
        unvisited.remove(best_c)
        total_demand += demands[best_c]
        
        d = dist_matrix[current, best_c]
        t += d
        b -= d
        current = best_c
        
        if t < ready[best_c]:
            t = ready[best_c]
        t += service[best_c]
    
    return route

def _solve_instance(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance.get("speed", 1.0)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]

    # Precompute distance matrix
    dist_matrix = _get_distance_matrix(coords)

    all_customers = list(range(1, n_customers + 1))
    start_time = time.time()
    main_deadline = start_time + 100  # Leave 50s for verification and fallback
    
    # Determine number of routes based on capacity
    total_demand = sum(demands[c] for c in all_customers)
    min_routes = max(1, (total_demand + capacity - 1) // capacity)
    n_clusters = min(n_customers, min_routes + 2)
    
    # Initial solution: Cluster + Nearest Neighbor
    clusters = _cluster_customers(coords, n_customers, n_clusters)
    
    routes = []
    for cluster in clusters:
        deadline_nn = min(time.time() + 5, main_deadline)
        route = _nearest_neighbor_route(
            cluster, dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_nn
        )
        if route:
            # Check feasibility
            feasible, _ = _check_route_feasibility_fast(
                dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, route,
                capacity, n_customers, stations
            )
            if feasible:
                routes.append(route)
            else:
                # Fallback: one customer per route for this cluster
                for c in cluster:
                    routes.append([c])
    
    # Ensure all customers are covered
    covered = set()
    for r in routes:
        for v in r:
            if v <= n_customers:
                covered.add(v)
    
    missing = [c for c in all_customers if c not in covered]
    for c in missing:
        routes.append([c])
    
    # Helper to get current demand of a route
    def _route_demand(route):
        return sum(demands[v] for v in route if v <= n_customers)

    # Apply 2-opt within each route
    deadline_2opt = min(time.time() + 25, main_deadline)
    for i in range(len(routes)):
        if time.time() > deadline_2opt:
            break
        routes[i], _ = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_2opt
        )
    
    # Apply Or-opt within each route
    deadline_oropt = min(time.time() + 20, main_deadline)
    for i in range(len(routes)):
        if time.time() > deadline_oropt:
            break
        routes[i], _ = _or_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_oropt
        )

    # DP-based route optimization for small routes
    deadline_dp = min(time.time() + 30, main_deadline)
    for i in range(len(routes)):
        if time.time() > deadline_dp:
            break
        route = routes[i]
        # Only apply DP for routes with <= 15 customers
        cust_in_route = [v for v in route if v <= n_customers]
        if len(cust_in_route) <= 15:
            dp_route, dp_dist = _dp_route_search(
                cust_in_route, dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity, n_customers, stations,
                deadline_dp
            )
            if dp_route is not None and dp_dist < _route_distance_fast(dist_matrix, route):
                routes[i] = dp_route
    
    # Local Search: Relocate customers between/within routes
    max_iterations = 60
    iteration = 0
    improved = True
    deadline_ls = min(time.time() + 35, main_deadline)
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time.time() > deadline_ls:
            break
        
        # Build customer-to-route index for speed
        cust_route = {}
        for i, r in enumerate(routes):
            for v in r:
                if v <= n_customers:
                    cust_route[v] = i
        
        # Iterate through all customers
        for c in all_customers:
            if time.time() > deadline_ls:
                break
            
            curr_route_idx = cust_route.get(c, -1)
            if curr_route_idx == -1:
                continue
            
            curr_route = routes[curr_route_idx]
            
            # Remove c from current route
            remaining_curr = [v for v in curr_route if v != c]
            
            best_target_idx = None
            best_improvement = 0.0
            best_new_routes_state = None
            
            # Calculate distance of current route without c
            dist_curr_without_c = _route_distance_fast(dist_matrix, remaining_curr) if remaining_curr else 0.0
            
            # Try inserting c into every other route (and current route at different pos)
            for i, route in enumerate(routes):
                if i == curr_route_idx:
                    target_route = remaining_curr
                    old_target_dist = dist_curr_without_c
                else:
                    target_route = route
                    old_target_dist = _route_distance_fast(dist_matrix, route)
                
                # Quick capacity check
                if _route_demand(target_route) + demands[c] > capacity:
                    continue
                
                best_pos = None
                best_target_dist = float('inf')
                
                for pos in range(len(target_route) + 1):
                    new_target_route = target_route[:pos] + [c] + target_route[pos:]
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_target_route,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < best_target_dist:
                        best_target_dist = nd
                        best_pos = pos
                
                if best_pos is not None:
                    if i == curr_route_idx:
                        old_affected = _route_distance_fast(dist_matrix, curr_route)
                        new_affected = best_target_dist
                        delta = new_affected - old_affected
                    else:
                        old_affected = _route_distance_fast(dist_matrix, curr_route) + old_target_dist
                        new_affected = dist_curr_without_c + best_target_dist
                        delta = new_affected - old_affected
                    
                    if delta < best_improvement:
                        best_improvement = delta
                        best_target_idx = i
                        new_routes = routes.copy()
                        
                        if i == curr_route_idx:
                            new_routes[curr_route_idx] = target_route[:best_pos] + [c] + target_route[best_pos:]
                        else:
                            new_routes[curr_route_idx] = remaining_curr
                            new_routes[i] = route[:best_pos] + [c] + route[best_pos:]
                        
                        best_new_routes_state = new_routes
            
            if best_target_idx is not None and best_improvement < -1e-6:
                routes = best_new_routes_state
                improved = True

    # Final 2-opt polish pass on each route
    deadline_polish = min(time.time() + 25, main_deadline)
    for i in range(len(routes)):
        if time.time() > deadline_polish:
            break
        routes[i], _ = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_polish
        )
    
    # Final Or-opt polish
    deadline_polish2 = min(time.time() + 15, main_deadline)
    for i in range(len(routes)):
        if time.time() > deadline_polish2:
            break
        routes[i], _ = _or_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_polish2
        )

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Solve
    routes = _solve_instance(instance)
    
    # Final Verification
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]
    
    dist_matrix = _get_distance_matrix(coords)
    
    all_ok = True
    served_customers = set()
    for r in routes:
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
        for v in r:
            if v <= n_customers:
                if v in served_customers:
                    all_ok = False
                    break
                served_customers.add(v)
        if not all_ok:
            break
    
    # Check all customers are served
    if all_ok:
        all_customers = set(range(1, n_customers + 1))
        if served_customers != all_customers:
            all_ok = False
    
    if not all_ok:
        # Fallback: One customer per route
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]
        
    return routes
# EVOLVE-BLOCK-END
