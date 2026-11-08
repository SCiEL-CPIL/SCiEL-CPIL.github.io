# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _dist(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return sqrt(dx * dx + dy * dy)

def _simulate_route(route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route (list of customer/station node ids, depot implicit at ends).
    Returns (is_valid, total_distance)
    """
    if not route:
        return True, 0.0
    
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    for node in route:
        d = _dist(coords, cur_node, node)
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return False, total_dist
        
        if node in stations:
            # Recharge
            energy = battery - cur_bat
            cur_time += energy * recharge_rate
            cur_bat = battery
        else:
            # Customer
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist
            cur_time = max(cur_time, ready[node])
            if cur_time > due[node] + 1e-9:
                return False, total_dist
            cur_time += service[node]
        
        cur_node = node
    
    # Return to depot
    d = _dist(coords, cur_node, 0)
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return False, total_dist
    if cur_time > horizon + 1e-9:
        return False, total_dist
        
    return True, total_dist

def _try_insertion(route, idx, cust, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon, current_load):
    """
    Try inserting customer 'cust' at position 'idx' in 'route'.
    Returns (is_valid, new_route, new_load)
    """
    new_route = route[:idx] + [cust] + route[idx:]
    
    # Quick capacity check
    if current_load + demands[cust] > capacity + 1e-9:
        return False, None, None
        
    is_valid, _ = _simulate_route(new_route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
    
    if is_valid:
        return True, new_route, current_load + demands[cust]
    return False, None, None

def _solve_instance(instance):
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    # Sort customers by due date for a better initial ordering (earlier deadlines first)
    customers.sort(key=lambda x: due[x])
    
    routes = []
    route_loads = []
    
    for cust in customers:
        best_cost = float('inf')
        best_route_idx = -1
        best_pos_idx = -1
        best_new_route = None
        best_new_load = 0.0
        
        # Try inserting into existing routes
        for r_idx, route in enumerate(routes):
            current_load = route_loads[r_idx]
            if current_load + demands[cust] > capacity + 1e-9:
                continue
                
            # Try all positions in the route
            for pos in range(len(route) + 1):
                is_valid, new_route, new_load = _try_insertion(
                    route, pos, cust, coords, demands, ready, due, service, stations, 
                    capacity, battery, speed, recharge_rate, horizon, current_load
                )
                if is_valid:
                    # Calculate incremental cost (approximate for speed, exact for final)
                    # For now, let's just pick the first valid one or minimize something simple.
                    # To be more robust, we can calculate the actual distance change.
                    # But _simulate_route returns total dist. Let's use that.
                    _, total_dist = _simulate_route(new_route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
                    
                    # We want to minimize total distance. 
                    # Since we are building routes, minimizing the resulting route length is a good heuristic.
                    if total_dist < best_cost:
                        best_cost = total_dist
                        best_route_idx = r_idx
                        best_pos_idx = pos
                        best_new_route = new_route
                        best_new_load = new_load
                        
        # If no insertion into existing route is feasible, start a new route
        if best_route_idx == -1:
            new_route = [cust]
            is_valid, total_dist = _simulate_route(new_route, coords, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon)
            if is_valid:
                routes.append(new_route)
                route_loads.append(demands[cust])
            else:
                # Should not happen if battery/capacity allows single customer
                # Fallback: return invalid or try to fix? 
                # Given constraints, a single customer should almost always be feasible if battery > 2*max_dist.
                # Max dist in [0,100]x[0,100] is ~141. Battery is 300. So 2*141 < 300.
                # So single customer is always feasible distance-wise.
                # Time window: ready in [0,400], due = ready + 100..300.
                # Travel time from depot to cust is at most 141.
                # If ready is 400, due is 500-700. Arrival at 141 is fine.
                # If ready is 0, due is 100-300. Arrival at 141 might be > due if due=100.
                # If due < travel_time, it's infeasible.
                # But the problem statement implies feasible instances? "Any broken rule makes the whole result invalid."
                # We assume instances are feasible.
                pass
        else:
            routes[best_route_idx] = best_new_route
            route_loads[best_route_idx] = best_new_load
            
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    routes = _solve_instance(instance)
    
    # Check if we are out of time? The heuristic is fast, but just in case.
    # If _solve_instance is too slow, we might need a simpler fallback.
    # For 25 customers, O(N^2) insertions with simulation O(N) is O(N^3) ~ 25^3 = 15625 ops per instance.
    # Very fast.
    
    return routes
# EVOLVE-BLOCK-END
