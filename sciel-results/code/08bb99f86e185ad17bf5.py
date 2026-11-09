"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _build_dist_matrix(coords):
    n = len(coords)
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist

def _simulate_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """
    Simulate a route to check feasibility and get metrics.
    Returns (is_feasible, total_dist, total_time).
    """
    if not route:
        return True, 0.0, 0.0
    
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0
        
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        time_t += d / speed
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0, 0.0
            time_t += service[v]
        else:
            # Charging station
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    # Return to depot
    d = dist[prev][0]
    battery -= d
    time_t += d / speed
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t

def _construct_savings(instance, dist):
    """
    Fast construction using Clarke-Wright Savings heuristic adapted for EV constraints.
    Returns a list of routes.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Initialize: each customer is in its own route
    # We need to handle stations. For construction, we'll ignore stations initially 
    # and insert them later if battery is an issue, or just try to serve customers directly.
    # Given battery 300 and coords in [0,100], max dist is ~141. Round trip ~282.
    # Most single customer routes are feasible without charging if close to depot.
    # If a route becomes infeasible due to battery, we might need to insert a station.
    
    # Start with individual routes
    routes = [[i] for i in range(1, n_customers + 1)]
    route_ids = list(range(1, n_customers + 1)) # Map: customer -> route index in 'routes' list? 
    # Better: Maintain a list of lists.
    
    # Calculate savings for merging route i and route j
    # Savings = dist[0][i] + dist[0][j] - dist[i][j]
    
    # To make this efficient, we just do a simple greedy merge.
    # Sort potential merges by savings.
    
    # For simplicity and speed, let's use a Nearest Neighbor approach first, then merge?
    # No, let's stick to a simplified Savings.
    
    # Actually, with 25 customers, a simple "insert into best existing route" is very fast.
    unserved = list(range(1, n_customers + 1))
    random.shuffle(unserved)
    
    active_routes = []
    route_demands = []
    
    # Pre-compute distances from depot to all nodes
    dist_depot = dist[0]
    
    while unserved:
        # Try to insert the next customer into an existing route
        c = unserved.pop(0)
        
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(active_routes):
            if route_demands[r_idx] + demands[c] > capacity:
                continue
                
            # Try inserting c into every position of this route
            # Optimization: Only check a few positions (start, end, middle, or neighbors)
            # For small routes, check all.
            n_r = len(route)
            if n_r < 8:
                positions = range(n_r + 1)
            else:
                # Check start, end, and a few random/middle spots
                positions = [0, n_r, n_r//2, max(0, n_r//2-1), min(n_r, n_r//2+1)]
                # Remove duplicates
                positions = list(set(positions))
                
            for pos in positions:
                # Calculate distance increment
                if pos == 0:
                    if n_r == 0:
                        inc = dist_depot[c] + dist_depot[c] # Wait, depot to c + c to depot? No, it's part of a route.
                        # Route is depot -> ... -> depot
                        # If pos 0: depot -> c -> route[0]
                        inc = dist_depot[c] + dist[c][route[0]] - dist_depot[route[0]]
                    else:
                        inc = dist_depot[c] + dist[c][route[0]] - dist_depot[route[0]]
                elif pos == n_r:
                    inc = dist[route[-1]][c] + dist_depot[c] - dist_depot[route[-1]]
                else:
                    inc = dist[route[pos-1]][c] + dist[c][route[pos]] - dist[route[pos-1]][route[pos]]
                    
                if inc >= best_cost:
                    continue
                    
                # Construct new route for feasibility check
                new_route = route[:pos] + [c] + route[pos:]
                
                # Fast feasibility check
                feasible, _, _ = _simulate_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                
                if feasible:
                    best_cost = inc
                    best_route_idx = r_idx
                    best_pos = pos
        
        if best_route_idx != -1:
            # Insert c into best_route_idx at best_pos
            route = active_routes[best_route_idx]
            active_routes[best_route_idx] = route[:best_pos] + [c] + route[best_pos:]
            route_demands[best_route_idx] += demands[c]
        else:
            # Start a new route
            # Check if single customer route is feasible
            new_route = [c]
            feasible, _, _ = _simulate_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible:
                active_routes.append(new_route)
                route_demands.append(demands[c])
            else:
                # This customer cannot be served alone? 
                # With battery 300 and max dist ~141, it should be possible unless time window is tight.
                # If not feasible, we might need to insert a station.
                # Let's try adding a station to the route.
                # Find the closest station to c or depot.
                stations = instance["stations"]
                best_station = None
                best_station_dist = float('inf')
                for s in stations:
                    d = dist[c][s]
                    if d < best_station_dist:
                        best_station_dist = d
                        best_station = s
                        
                if best_station is not None:
                    # Try route: c -> station -> depot? Or depot -> station -> c -> depot?
                    # Let's try depot -> station -> c -> depot
                    new_route_with_station = [best_station, c]
                    feasible, _, _ = _simulate_route(new_route_with_station, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if not feasible:
                         # Try c -> station -> depot
                         new_route_with_station = [c, best_station]
                         feasible, _, _ = _simulate_route(new_route_with_station, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                         
                    if feasible:
                        active_routes.append(new_route_with_station)
                        route_demands.append(demands[c])
                    else:
                        # Last resort: just append, we will fix in local search or it will be invalid.
                        # But we must return valid routes.
                        # If we can't serve it, we have a problem. 
                        # Given constraints, this is unlikely.
                        active_routes.append(new_route)
                        route_demands.append(demands[c])

    return active_routes

def _local_search(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """
    Apply 2-Opt and Or-Opt moves.
    """
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        for r_idx in range(len(routes)):
            if time.time() > deadline - 1:
                return routes
            route = routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            
            # 2-Opt
            found_improvement = False
            for i in range(n - 1):
                if time.time() > deadline - 1:
                    return routes
                for j in range(i + 1, n):
                    # Quick distance check
                    prev_i = route[i-1] if i > 0 else 0
                    next_j = route[j+1] if j < n - 1 else 0
                    
                    d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                    d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                    
                    if d_new < d_old - 1e-9:
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        feasible, _, _ = _simulate_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            routes[r_idx] = new_route
                            found_improvement = True
                            improved = True
                            break
                if found_improvement:
                    break
            
            # Or-Opt (Move 1 node)
            if not found_improvement and time.time() < deadline - 1:
                n = len(routes[r_idx])
                for i in range(n):
                    if time.time() > deadline - 1:
                        return routes
                    node = routes[r_idx][i]
                    rest = routes[r_idx][:i] + routes[r_idx][i+1:]
                    
                    best_pos = -1
                    best_cost = float('inf')
                    
                    # Calculate cost of removing node
                    if i == 0:
                        prev_i = 0
                        next_i = rest[0] if len(rest) > 0 else 0
                        d_remove = dist[prev_i][node] + dist[node][next_i] - dist[prev_i][next_i]
                    elif i == len(routes[r_idx]) - 1:
                        prev_i = rest[-1] if len(rest) > 0 else 0
                        next_i = 0
                        d_remove = dist[prev_i][node] + dist[node][next_i] - dist[prev_i][next_i]
                    else:
                        prev_i = routes[r_idx][i-1]
                        next_i = routes[r_idx][i+1]
                        d_remove = dist[prev_i][node] + dist[node][next_i] - dist[prev_i][next_i]
                        
                    for pos in range(len(rest) + 1):
                        # Skip original position
                        if pos == i or (pos == i - 1 and i > 0):
                            continue
                            
                        if pos == 0:
                            d_insert = dist[0][node] + dist[node][rest[0]] - dist[0][rest[0]] if len(rest)>0 else dist[0][node]*2
                        elif pos == len(rest):
                            d_insert = dist[rest[-1]][node] + dist[node][0] - dist[rest[-1]][0]
                        else:
                            d_insert = dist[rest[pos-1]][node] + dist[node][rest[pos]] - dist[rest[pos-1]][rest[pos]]
                            
                        delta = d_insert - d_remove
                        if delta < -1e-9 and delta < best_cost:
                            best_cost = delta
                            best_pos = pos
                            
                    if best_pos != -1:
                        new_route = rest[:best_pos] + [node] + rest[best_pos:]
                        feasible, _, _ = _simulate_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            routes[r_idx] = new_route
                            improved = True
                            break # Restart or-opt for this route
        
    return routes

def plan(instance):
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    # Leave 5 seconds buffer for final checks and return
    deadline = start_time + 115 
    
    coords = instance["coords"]
    dist = _build_dist_matrix(coords)
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # 1. Construction
    routes = _construct_savings(instance, dist)
    
    # 2. Local Search
    routes = _local_search(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline)
    
    # 3. Final Verification and Cleanup
    # Ensure all customers are served exactly once
    served = set()
    valid_routes = []
    for r in routes:
        # Check for duplicates or invalid nodes
        if not r:
            continue
        feasible, _, _ = _simulate_route(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            # Check for duplicates
            for node in r:
                if node in served:
                    feasible = False
                    break
            if feasible:
                served.update(r)
                valid_routes.append(r)
    
    # If some customers are missing, add them as single-node routes (best effort)
    missing = [c for c in range(1, n_customers + 1) if c not in served]
    for c in missing:
        valid_routes.append([c])
        
    return valid_routes
# EVOLVE-BLOCK-END
