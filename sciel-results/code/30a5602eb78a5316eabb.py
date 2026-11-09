"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _get_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    n = len(coords)
    arr = np.array(coords, dtype=np.float64)
    dx = arr[:, 0, None] - arr[None, :, 0]
    dy = arr[:, 1, None] - arr[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2)
    return dist

def _route_distance(dist_matrix, route):
    """Calculate total distance of a route using precomputed distance matrix."""
    if not route:
        return 0.0
    d = dist_matrix[0, route[0]]
    for k in range(len(route) - 1):
        d += dist_matrix[route[k], route[k + 1]]
    d += dist_matrix[route[-1], 0]
    return d

def _check_route_feasibility(dist_matrix, demands, ready, due, service,
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
    b = float(battery)
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
            b = float(battery)  # Refill to full
        
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

def _two_opt_route(route, dist_matrix, demands, ready, due, service,
                   battery, recharge_rate, horizon, capacity, n_customers, stations,
                   deadline):
    """Apply 2-opt moves within a single route to reduce distance while maintaining feasibility."""
    if len(route) < 3:
        return route
    
    n = len(route)
    improved = True
    
    while improved:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                a = 0 if i == 0 else route[i - 1]
                b_node = route[i]
                c = route[j]
                d_node = 0 if j == n - 1 else route[j + 1]
                
                old_d = dist_matrix[a, b_node] + dist_matrix[c, d_node]
                new_d = dist_matrix[a, c] + dist_matrix[b_node, d_node]
                
                if new_d >= old_d - 1e-9:
                    continue
                
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                feasible, _ = _check_route_feasibility(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                
                if feasible:
                    route = new_route
                    improved = True
    
    return route

def _or_opt_route(route, dist_matrix, demands, ready, due, service,
                  battery, recharge_rate, horizon, capacity, n_customers, stations,
                  deadline):
    """Apply Or-opt moves (relocate 1-3 consecutive nodes) within a route."""
    if len(route) < 4:
        return route
    
    n = len(route)
    improved = True
    
    while improved:
        improved = False
        for seg_len in [1, 2, 3]:
            if time.time() > deadline:
                return route
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    return route
                segment = route[i:i+seg_len]
                remaining = route[:i] + route[i+seg_len:]
                
                # Try inserting segment at every position in remaining
                for pos in range(len(remaining) + 1):
                    if pos == i or pos == i + seg_len:
                        continue
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    
                    feasible, nd = _check_route_feasibility(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_route,
                        capacity, n_customers, stations
                    )
                    old_d = _route_distance(dist_matrix, route)
                    if feasible and nd < old_d - 1e-9:
                        route = new_route
                        n = len(route)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    
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
    
    # Initial solution: Greedy insertion sorted by due date
    routes = []
    unassigned = sorted(all_customers, key=lambda c: due[c])
    
    def _route_demand(route):
        return sum(demands[v] for v in route if v <= n_customers)

    for c in unassigned:
        best_route_idx = None
        best_dist_increase = float('inf')
        best_new_route = None
        
        for i, route in enumerate(routes):
            if _route_demand(route) + demands[c] > capacity:
                continue
                
            best_pos_in_route = None
            best_route_dist = float('inf')
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, nd = _check_route_feasibility(
                    dist_matrix, demands, ready, due, service,
                    battery, recharge_rate, horizon, new_route,
                    capacity, n_customers, stations
                )
                if feasible and nd < best_route_dist:
                    best_route_dist = nd
                    best_pos_in_route = pos
            
            if best_pos_in_route is not None:
                old_route_dist = _route_distance(dist_matrix, route)
                increase = best_route_dist - old_route_dist
                if increase < best_dist_increase:
                    best_dist_increase = increase
                    best_route_idx = i
                    best_new_route = route[:best_pos_in_route] + [c] + route[best_pos_in_route:]
        
        if best_route_idx is not None:
            routes[best_route_idx] = best_new_route
        else:
            routes.append([c])

    # Apply 2-opt and Or-opt within each route
    deadline_2opt = time.time() + 15
    for i in range(len(routes)):
        if time.time() > deadline_2opt:
            break
        routes[i] = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_2opt
        )
        routes[i] = _or_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_2opt
        )

    # Local Search: Relocate customers between/within routes
    deadline_ls = time.time() + 35
    improved = True
    iteration = 0
    max_iterations = 60
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        if time.time() > deadline_ls:
            break
        
        cust_route = {}
        for i, r in enumerate(routes):
            for v in r:
                cust_route[v] = i
        
        for c in all_customers:
            if time.time() > deadline_ls:
                break
            
            curr_route_idx = cust_route.get(c, -1)
            if curr_route_idx == -1:
                continue
            
            curr_route = routes[curr_route_idx]
            remaining_curr = [v for v in curr_route if v != c]
            
            best_target_idx = None
            best_improvement = 0.0
            best_new_routes_state = None
            
            dist_curr_without_c = _route_distance(dist_matrix, remaining_curr) if remaining_curr else 0.0
            
            for i, route in enumerate(routes):
                if i == curr_route_idx:
                    target_route = remaining_curr
                    old_target_dist = dist_curr_without_c
                else:
                    target_route = route
                    old_target_dist = _route_distance(dist_matrix, route)
                
                if _route_demand(target_route) + demands[c] > capacity:
                    continue
                
                best_pos = None
                best_target_dist = float('inf')
                
                for pos in range(len(target_route) + 1):
                    new_target_route = target_route[:pos] + [c] + target_route[pos:]
                    feasible, nd = _check_route_feasibility(
                        dist_matrix, demands, ready, due, service,
                        battery, recharge_rate, horizon, new_target_route,
                        capacity, n_customers, stations
                    )
                    if feasible and nd < best_target_dist:
                        best_target_dist = nd
                        best_pos = pos
                
                if best_pos is not None:
                    if i == curr_route_idx:
                        old_affected = _route_distance(dist_matrix, curr_route)
                        new_affected = best_target_dist
                        delta = new_affected - old_affected
                    else:
                        old_affected = _route_distance(dist_matrix, curr_route) + old_target_dist
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

    # Final 2-opt and Or-opt polish
    deadline_polish = time.time() + 20
    for i in range(len(routes)):
        if time.time() > deadline_polish:
            break
        routes[i] = _two_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_polish
        )
        routes[i] = _or_opt_route(
            routes[i], dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, capacity, n_customers, stations,
            deadline_polish
        )

    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 130  # 20s margin
    
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
        ok, _ = _check_route_feasibility(
            dist_matrix, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break
            
    if not all_ok:
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]
        
    return routes
# EVOLVE-BLOCK-END
