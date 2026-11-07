import numpy as np
import time
import random
import math

DEADLINE_SECONDS = 145.0  # Safety margin below 150s limit

def _build_dist_matrix(coords):
    """Precompute distance matrix using vectorized operations."""
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff * diff, axis=2))
    return dist_matrix

def _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """
    Simulate a route. Returns (valid, distance, time_at_depot, battery_at_depot).
    route: list of node IDs (1..n_customers or station IDs). Depot is 0.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    # Check demand capacity
    total_demand = 0
    for node in route:
        if node <= n_customers:
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
        
        if node <= n_customers:
            # Customer
            time_at = max(time_at, ready[node])
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations_set:
            # Station
            energy_needed = battery_max - battery
            if energy_needed > 1e-9:
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
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    
    stations_set = set(instance["stations"])
    served = set()
    total_distance = 0.0
    
    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                                             instance["battery"], instance["speed"],
                                             instance["recharge_rate"], instance["horizon"],
                                             instance["capacity"])
        if not valid:
            return False, 0.0
        total_distance += dist
        
        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)
    
    if served != set(range(1, n_customers + 1)):
        return False, 0.0
    
    return True, total_distance

def _compute_total_distance(routes, dist_matrix):
    """Compute total distance of routes."""
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total

def _check_route_feasibility(route, dist_matrix, instance, n_customers, stations_set,
                              battery_max, speed, recharge_rate, horizon, capacity):
    """Check if a route is feasible."""
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                                      battery_max, speed, recharge_rate, horizon, capacity)
    return valid

def _greedy_construct(instance, dist_matrix):
    """Construct initial solution using greedy insertion with spatial clustering."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])
    
    # Sort customers by due time
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
                
                if _check_route_feasibility(new_route, dist_matrix, instance, n_customers, stations_set,
                                             battery_max, speed, recharge_rate, horizon, capacity):
                    # Approximate cost: distance change
                    # Optimization: Only calculate full distance if necessary
                    # For now, use full calculation for safety, but this is O(N^2)
                    # We can speed this up by precomputing route distances
                    old_dist = _compute_total_distance([route], dist_matrix)
                    new_dist = _compute_total_distance([new_route], dist_matrix)
                    
                    cost = new_dist - old_dist
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_insert_pos = pos
        
        if best_route_idx == -1:
            # Try to insert with a station if direct insertion fails?
            # For initial construction, just start a new route
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_insert_pos, cust)
            route_loads[best_route_idx] += demands[cust]
    
    return routes

def _optimize_route_with_stations(route, dist_matrix, instance, n_customers, stations_set,
                                   battery_max, speed, recharge_rate, horizon, capacity):
    """
    Attempt to insert charging stations into a route to make it feasible if it isn't,
    or to optimize existing stations.
    Returns the optimized route if feasible, else None.
    """
    # This is a heuristic repair/optimization
    # If the route is already feasible, try to remove unnecessary stations
    # If not feasible, try to add stations
    
    current_route = route[:]
    
    # Check if feasible
    if _check_route_feasibility(current_route, dist_matrix, instance, n_customers, stations_set,
                                 battery_max, speed, recharge_rate, horizon, capacity):
        # Try to remove stations to save time/distance? 
        # Actually, removing stations increases battery drain, so it might break feasibility.
        # We only want to remove stations if the route is still feasible without them.
        for i in range(len(current_route) - 1, -1, -1):
            if current_route[i] in stations_set:
                test_route = current_route[:i] + current_route[i+1:]
                if _check_route_feasibility(test_route, dist_matrix, instance, n_customers, stations_set,
                                             battery_max, speed, recharge_rate, horizon, capacity):
                    current_route = test_route
        return current_route
    
    # If not feasible, try inserting stations
    # Simple strategy: Insert a station after the customer where battery drops lowest
    # Or just try inserting a station at every possible position and see if it helps
    
    best_route = None
    best_dist = float('inf')
    
    # Get list of customers in route
    customers_in_route = [n for n in current_route if n <= n_customers]
    
    # Try inserting a station after each customer
    for cust_idx, cust in enumerate(customers_in_route):
        # Find position of this customer in current_route
        pos = current_route.index(cust)
        # Try inserting station after pos
        for station in instance["stations"]:
            new_route = current_route[:pos+1] + [station] + current_route[pos+1:]
            if _check_route_feasibility(new_route, dist_matrix, instance, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity):
                dist = _compute_total_distance([new_route], dist_matrix)
                if dist < best_dist:
                    best_dist = dist
                    best_route = new_route
    
    if best_route:
        return best_route
    
    return None

def _local_search(routes, dist_matrix, instance, deadline):
    """
    Perform local search: 2-opt, Or-opt, Relocate, Swap.
    """
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])
    
    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)
    
    def is_feasible(route):
        return _check_route_feasibility(route, dist_matrix, instance, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity)
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    
    improved = True
    max_passes = 10
    pass_count = 0
    
    while improved and pass_count < max_passes:
        improved = False
        pass_count += 1
        
        # 1. Intra-route 2-opt
        for r_idx in range(len(current_routes)):
            route = current_routes[r_idx]
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_feasible(new_route):
                        new_dist = get_total_dist([new_route])
                        old_dist = get_total_dist([route])
                        if new_dist < old_dist - 1e-9:
                            current_routes[r_idx] = new_route
                            current_dist = get_total_dist(current_routes)
                            improved = True
        
        # 2. Inter-route Relocate
        for r1 in range(len(current_routes)):
            for pos1 in range(len(current_routes[r1])):
                cust = current_routes[r1][pos1]
                if cust > n_customers: # Skip stations for now in relocate
                    continue
                
                for r2 in range(len(current_routes)):
                    if r1 == r2:
                        continue
                    if route_load(current_routes[r2]) + demands[cust] > capacity:
                        continue
                    
                    for pos2 in range(len(current_routes[r2]) + 1):
                        route1_new = current_routes[r1][:pos1] + current_routes[r1][pos1+1:]
                        route2_new = current_routes[r2][:pos2] + [cust] + current_routes[r2][pos2:]
                        
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in current_routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                current_routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # 3. Inter-route Swap
        if not improved:
            for r1 in range(len(current_routes)):
                for pos1 in range(len(current_routes[r1])):
                    cust1 = current_routes[r1][pos1]
                    if cust1 > n_customers:
                        continue
                    for r2 in range(len(current_routes)):
                        if r1 == r2:
                            continue
                        for pos2 in range(len(current_routes[r2])):
                            cust2 = current_routes[r2][pos2]
                            if cust2 > n_customers:
                                continue
                            
                            load1 = route_load(current_routes[r1])
                            load2 = route_load(current_routes[r2])
                            if load1 - demands[cust1] + demands[cust2] > capacity:
                                continue
                            if load2 - demands[cust2] + demands[cust1] > capacity:
                                continue
                            
                            route1_new = current_routes[r1][:]
                            route1_new[pos1] = cust2
                            route2_new = current_routes[r2][:]
                            route2_new[pos2] = cust1
                            
                            if is_feasible(route1_new) and is_feasible(route2_new):
                                new_routes = [r[:] for r in current_routes]
                                new_routes[r1] = route1_new
                                new_routes[r2] = route2_new
                                new_dist = get_total_dist(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    current_routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        # Remove empty routes
        current_routes = [r for r in current_routes if len(r) > 0]
        
        if time.time() > deadline:
            break

    return current_routes

def _simulated_annealing(routes, dist_matrix, instance, deadline):
    """
    Simulated Annealing with diverse moves including station insertion/removal.
    """
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_list = instance["stations"]
    stations_set = set(stations_list)
    
    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)
    
    def is_feasible(route):
        return _check_route_feasibility(route, dist_matrix, instance, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity)
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist
    
    start_temp = 5.0
    end_temp = 0.01
    # Estimate number of iterations we can do
    # Assume ~1000 iterations per second
    time_left = deadline - time.time()
    max_iters = int(time_left * 800) 
    max_iters = max(100, min(max_iters, 10000))
    
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp
    
    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        new_routes = [r[:] for r in current_routes]
        
        move_type = random.random()
        
        if move_type < 0.3:
            # Move customer (Relocate)
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            if len(new_routes[r2]) > 0:
                                pos2 = random.randint(0, len(new_routes[r2]))
                                new_routes[r2].insert(pos2, cust)
                            else:
                                new_routes[r2].append(cust)
        
        elif move_type < 0.5:
            # Swap customers
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    
                    if cust1 <= n_customers and cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        
        elif move_type < 0.7:
            # 2-opt within a route
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]
        
        elif move_type < 0.85:
            # Insert Station
            if len(new_routes) > 0 and len(stations_list) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos = random.randint(0, len(new_routes[r1]))
                    station = random.choice(stations_list)
                    # Avoid duplicate station at same pos
                    if pos < len(new_routes[r1]) and new_routes[r1][pos] == station:
                        continue
                    new_routes[r1].insert(pos, station)
        
        else:
            # Remove Station
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                # Find a station in this route
                station_pos = -1
                for i, node in enumerate(new_routes[r1]):
                    if node in stations_set:
                        station_pos = i
                        break
                if station_pos != -1:
                    new_routes[r1].pop(station_pos)
        
        # Filter empty routes
        new_routes = [r for r in new_routes if len(r) > 0]
        
        # Check feasibility
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if node <= n_customers:
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
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Build distance matrix once
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    
    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    # 1. Greedy Construction
    routes = _greedy_construct(instance, dist_matrix)
    
    # Validate initial
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        # Fallback: single customer per route
        customers = list(range(1, n_customers + 1))
        routes = [[v] for v in customers]
    
    # 2. Local Search (Fast improvement)
    if time.time() < deadline - 10:
        routes = _local_search(routes, dist_matrix, instance, deadline - 10)
        valid, dist = _validate_solution(routes, instance)
        if not valid:
            customers = list(range(1, n_customers + 1))
            routes = [[v] for v in customers]
    
    # 3. Simulated Annealing (Deep optimization)
    if time.time() < deadline - 5:
        sa_routes = _simulated_annealing(routes, dist_matrix, instance, deadline - 5)
        valid_sa, dist_sa = _validate_solution(sa_routes, instance)
        if valid_sa and dist_sa < dist:
            routes = sa_routes
            dist = dist_sa
    
    # Final validation
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        # Last resort fallback
        customers = list(range(1, n_customers + 1))
        routes = [[v] for v in customers]
    
    return routes
