"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt

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
        d = dist_matrix[prev][v]
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
    d = dist_matrix[prev][0]
    t += d
    b -= d
    total_dist += d
    if b < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, total_dist

def _kmeans_cluster(coords, n_customers, n_clusters, n_iter=30, seed=42):
    """Simple k-means clustering of customers (nodes 1..n_customers)."""
    rng = random.Random(seed)
    cust_coords = np.array([coords[i] for i in range(1, n_customers + 1)])
    
    # Initialize centroids by picking random customers
    idx = list(range(n_customers))
    rng.shuffle(idx)
    centroids = [cust_coords[idx[i % n_customers]] for i in range(n_clusters)]
    
    labels = np.zeros(n_customers, dtype=int)
    
    for _ in range(n_iter):
        # Assign each customer to nearest centroid
        for i in range(n_customers):
            best_c = 0
            best_d = float('inf')
            for c in range(n_clusters):
                d = (cust_coords[i][0] - centroids[c][0])**2 + (cust_coords[i][1] - centroids[c][1])**2
                if d < best_d:
                    best_d = d
                    best_c = c
            labels[i] = best_c
        
        # Update centroids
        for c in range(n_clusters):
            members = [i for i in range(n_customers) if labels[i] == c]
            if members:
                centroids[c] = cust_coords[members].mean(axis=0)
    
    return labels

def _nearest_neighbor_route(customers, dist_matrix, demands, ready, due, service,
                           battery, recharge_rate, horizon, capacity, n_customers, stations):
    """Build a route using nearest neighbor heuristic with time window awareness."""
    route = []
    unvisited = set(customers)
    t = 0.0
    b = battery
    prev = 0
    total_demand = 0
    
    while unvisited:
        best_node = None
        best_score = float('inf')
        
        for v in unvisited:
            if total_demand + demands[v] > capacity:
                continue
            
            d = dist_matrix[prev][v]
            new_t = t + d
            new_b = b - d
            
            if new_b < -1e-9:
                continue
            
            if new_t < ready[v]:
                new_t = ready[v]
            
            if new_t > due[v] + 1e-9:
                continue
            
            # Score: distance + small penalty for waiting
            score = d + 0.1 * max(0, ready[v] - (t + d))
            
            if score < best_score:
                best_score = score
                best_node = v
        
        if best_node is None:
            # No feasible insertion, pick the one with earliest due
            best_node = min(unvisited, key=lambda v: due[v])
        
        route.append(best_node)
        unvisited.discard(best_node)
        total_demand += demands[best_node]
        
        d = dist_matrix[prev][best_node]
        t += d
        b -= d
        
        if t < ready[best_node]:
            t = ready[best_node]
        t += service[best_node]
        prev = best_node
    
    return route

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
                a = 0 if i == 0 else route[i - 1]
                b_node = route[i]
                c = route[j]
                d_node = 0 if j == n - 1 else route[j + 1]
                
                old_d = dist_matrix[a][b_node] + dist_matrix[c][d_node]
                new_d = dist_matrix[a][c] + dist_matrix[b_node][d_node]
                
                if new_d >= old_d - 1e-9:
                    continue
                
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                feasible, _ = _check_route_feasibility_fast(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                
                if feasible:
                    route = new_route
                    total_improvement += old_d - new_d
                    improved = True
    
    return route, total_improvement

def _or_opt_route(route, dist_matrix, demands, ready, due, service,
                  battery, recharge_rate, horizon, capacity, n_customers, stations,
                  deadline):
    """Apply Or-opt moves (relocate 1-3 consecutive nodes) within a route."""
    if len(route) < 3:
        return route, 0.0
    
    n = len(route)
    improved = True
    total_improvement = 0.0
    max_seg = min(3, n - 1)
    
    while improved:
        improved = False
        for seg_len in range(1, max_seg + 1):
            if time.time() > deadline:
                return route, total_improvement
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    return route, total_improvement
                segment = route[i:i+seg_len]
                remaining = route[:i] + route[i+seg_len:]
                
                # Try inserting segment at every position in remaining
                for pos in range(len(remaining) + 1):
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    if new_route == route:
                        continue
                    
                    old_d = _route_distance_fast(dist_matrix, route)
                    new_d = _route_distance_fast(dist_matrix, new_route)
                    
                    if new_d < old_d - 1e-9:
                        feasible, _ = _check_route_feasibility_fast(
                            dist_matrix, demands, ready, due, service,
                            battery, recharge_rate, horizon, new_route,
                            capacity, n_customers, stations
                        )
                        if feasible:
                            route = new_route
                            total_improvement += old_d - new_d
                            improved = True
                            n = len(route)
                            break
                if improved:
                    break
            if improved:
                break
    
    return route, total_improvement

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
    
    start_time = time.time()
    deadline = start_time + 130  # Safety margin

    # Precompute distance matrix
    dist_matrix = _get_distance_matrix(coords)

    all_customers = list(range(1, n_customers + 1))
    
    # Determine number of clusters/routes based on total demand
    total_demand = sum(demands[c] for c in all_customers)
    n_routes = max(1, int(np.ceil(total_demand / capacity)))
    n_routes = min(n_routes, n_customers)
    
    # K-means clustering for initial route assignment
    labels = _kmeans_cluster(coords, n_customers, n_routes)
    
    # Build initial routes from clusters
    routes = []
    for c in range(n_routes):
        cluster_customers = [i + 1 for i in range(n_customers) if labels[i] == c]
        if cluster_customers:
            route = _nearest_neighbor_route(
                cluster_customers, dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity, n_customers, stations
            )
            routes.append(route)
    
    # Check if all customers are assigned
    assigned = set()
    for r in routes:
        for v in r:
            assigned.add(v)
    missing = set(all_customers) - assigned
    if missing:
        # Add missing customers one by one
        for c in sorted(missing, key=lambda x: due[x]):
            best_route_idx = None
            best_increase = float('inf')
            
            for i, route in enumerate(routes):
                rd = sum(demands[v] for v in route if v <= n_customers)
                if rd + demands[c] > capacity:
                    continue
                
                old_dist = _route_distance_fast(dist_matrix, route)
                best_pos = None
                best_new_dist = float('inf')
                
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < best_new_dist:
                        best_new_dist = nd
                        best_pos = pos
                
                if best_pos is not None:
                    inc = best_new_dist - old_dist
                    if inc < best_increase:
                        best_increase = inc
                        best_route_idx = i
                        routes[i] = route[:best_pos] + [c] + route[best_pos:]
            
            if best_route_idx is None:
                routes.append([c])
    
    # Phase 1: 2-opt within each route
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i], _ = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline
        )
    
    # Phase 2: Or-opt within each route
    for i in range(len(routes)):
        if time.time() > deadline:
            break
        routes[i], _ = _or_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline
        )
    
    # Phase 3: Relocate and swap local search
    def _route_demand(route):
        return sum(demands[v] for v in route if v <= n_customers)
    
    ls_deadline = min(deadline, time.time() + 60)
    max_ls_iters = 100
    improved = True
    ls_iter = 0
    
    while improved and ls_iter < max_ls_iters:
        improved = False
        ls_iter += 1
        
        if time.time() > ls_deadline:
            break
        
        # Build customer-to-route index
        cust_route = {}
        for i, r in enumerate(routes):
            for v in r:
                if v <= n_customers:
                    cust_route[v] = i
        
        for c in all_customers:
            if time.time() > ls_deadline:
                break
            
            curr_idx = cust_route.get(c, -1)
            if curr_idx == -1:
                continue
            
            curr_route = routes[curr_idx]
            remaining_curr = [v for v in curr_route if v != c]
            dist_curr_without_c = _route_distance_fast(dist_matrix, remaining_curr) if remaining_curr else 0.0
            
            best_delta = 0.0
            best_action = None  # (type, idx, pos)
            
            # Try relocate to each route
            for i, route in enumerate(routes):
                if i == curr_idx:
                    target = remaining_curr
                    old_target_dist = dist_curr_without_c
                else:
                    target = route
                    old_target_dist = _route_distance_fast(dist_matrix, route)
                
                rd = _route_demand(target)
                if rd + demands[c] > capacity:
                    continue
                
                for pos in range(len(target) + 1):
                    new_target = target[:pos] + [c] + target[pos:]
                    feasible, nd = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_target,
                        capacity, n_customers, stations
                    )
                    if not feasible:
                        continue
                    
                    if i == curr_idx:
                        old_aff = _route_distance_fast(dist_matrix, curr_route)
                        new_aff = nd
                    else:
                        old_aff = _route_distance_fast(dist_matrix, curr_route) + old_target_dist
                        new_aff = dist_curr_without_c + nd
                    
                    delta = new_aff - old_aff
                    if delta < best_delta:
                        best_delta = delta
                        best_action = ('relocate', i, pos)
            
            # Try swap with a customer in another route
            for i, route in enumerate(routes):
                if i == curr_idx:
                    continue
                if time.time() > ls_deadline:
                    break
                
                for j, other in enumerate(route):
                    if other > n_customers:
                        continue
                    
                    # Check capacity constraints for swap
                    curr_rd = _route_demand(remaining_curr) + demands[other]
                    other_rd = _route_demand(route) - demands[other] + demands[c]
                    if curr_rd > capacity or other_rd > capacity:
                        continue
                    
                    # Build new routes
                    new_curr = remaining_curr + [other]
                    new_other = [v for k, v in enumerate(route) if v != other] + [c]
                    
                    # Find best insertion positions
                    feasible_curr, dist_curr_new = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_curr,
                        capacity, n_customers, stations
                    )
                    feasible_other, dist_other_new = _check_route_feasibility_fast(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_other,
                        capacity, n_customers, stations
                    )
                    
                    if not feasible_curr or not feasible_other:
                        continue
                    
                    old_aff = _route_distance_fast(dist_matrix, curr_route) + _route_distance_fast(dist_matrix, route)
                    new_aff = dist_curr_new + dist_other_new
                    delta = new_aff - old_aff
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_action = ('swap', i, other)
            
            if best_action is not None and best_delta < -1e-6:
                if best_action[0] == 'relocate':
                    i, pos = best_action[1], best_action[2]
                    if i == curr_idx:
                        target = remaining_curr
                        routes[curr_idx] = target[:pos] + [c] + target[pos:]
                    else:
                        routes[curr_idx] = remaining_curr
                        target = routes[i]
                        routes[i] = target[:pos] + [c] + target[pos:]
                else:  # swap
                    i, other = best_action[1], best_action[2]
                    new_curr = [v for v in routes[curr_idx] if v != c] + [other]
                    new_other = [v for v in routes[i] if v != other] + [c]
                    routes[curr_idx] = new_curr
                    routes[i] = new_other
                
                improved = True
    
    # Phase 4: Final 2-opt + Or-opt polish
    if time.time() < deadline - 5:
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            routes[i], _ = _two_opt_route(
                routes[i], dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity, n_customers, stations,
                deadline
            )
    
    if time.time() < deadline - 2:
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            routes[i], _ = _or_opt_route(
                routes[i], dist_matrix, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity, n_customers, stations,
                deadline
            )

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
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
    for r in routes:
        ok, _ = _check_route_feasibility_fast(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
    
    # Check all customers served exactly once
    served = []
    for r in routes:
        for v in r:
            if v <= n_customers:
                served.append(v)
    
    if len(served) != n_customers or len(set(served)) != n_customers:
        all_ok = False
    
    if not all_ok:
        # Fallback: One customer per route
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]
        
    return routes
# EVOLVE-BLOCK-END
