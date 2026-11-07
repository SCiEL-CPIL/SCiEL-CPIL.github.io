# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _dist(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return math.hypot(dx, dy)

def _simulate(route, instance, coords):
    """
    Simulate a route starting at depot (node 0) with full battery.
    Returns (valid, total_distance, end_time)
    """
    n = len(route)
    if n == 0:
        return True, 0.0, 0.0
    
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    current_pos = 0
    time_t = 0.0
    battery = battery_max
    load = 0
    total_dist = 0.0
    
    for node in route:
        d = _dist(coords, current_pos, node)
        time_t += d / speed
        battery -= d
        total_dist += d
        
        if battery < -1e-9:
            return False, 0.0, 0.0
        
        # Check if station
        if node in instance["stations_set"]:
            # Recharge
            time_t += (battery_max - battery) * recharge_rate
            battery = battery_max
            # No demand, no service time for stations
        else:
            # Customer
            load += instance["demands"][node]
            if load > capacity:
                return False, 0.0, 0.0
            
            ready = instance["ready"][node]
            due = instance["due"][node]
            service = instance["service"][node]
            
            if time_t < ready:
                time_t = ready
            
            if time_t > due + 1e-9:
                return False, 0.0, 0.0
            
            time_t += service
        
        current_pos = node
    
    # Return to depot
    d = _dist(coords, current_pos, 0)
    time_t += d / speed
    battery -= d
    total_dist += d
    
    if battery < -1e-9:
        return False, 0.0, 0.0
    if time_t > horizon + 1e-9:
        return False, 0.0, 0.0
    
    return True, total_dist, time_t

def _constructive_route(instance, coords, unassigned, start_time=0.0, start_battery=300.0):
    """
    Greedy nearest neighbor construction for a single route.
    Returns (route, end_time, end_battery) or (None, None, None) if no customer can be added.
    """
    if not unassigned:
        return [], start_time, start_battery
    
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    current_pos = 0
    time_t = start_time
    battery = start_battery
    load = 0
    route = []
    remaining = list(unassigned)
    
    while remaining:
        best_node = -1
        best_dist = float('inf')
        
        for node in remaining:
            d = _dist(coords, current_pos, node)
            
            # Check battery to reach
            if battery - d < -1e-9:
                continue
            
            # Check capacity
            if load + instance["demands"][node] > capacity:
                continue
            
            # Check time window
            t_arrive = time_t + d / speed
            ready = instance["ready"][node]
            due = instance["due"][node]
            service = instance["service"][node]
            
            t_service_start = max(t_arrive, ready)
            if t_service_start > due + 1e-9:
                continue
            
            # Check if we can return to depot after service
            # (Rough check: battery after service + return)
            t_after_service = t_service_start + service
            d_return = _dist(coords, node, 0)
            if battery - d - d_return < -1e-9:
                # Maybe we need to go to a station first? For simplicity in constructive, 
                # if we can't return directly, skip for now or handle later.
                # Actually, if we can't return directly, we might still be able to go to a station.
                # But for NN greedy, let's just check direct return feasibility for simplicity of insertion.
                # A more sophisticated check would try to find a feasible path to depot via stations.
                # For now, if direct return is not possible, we might still insert if we can go to a station.
                # Let's allow it if we can at least reach the node. The simulator will catch infeasibility later.
                # But to avoid creating obviously bad routes, let's be conservative.
                # However, with battery 300 and coords in [0,100], max dist is ~141.
                # So battery is rarely the constraint unless many long legs.
                pass
            
            if d < best_dist:
                best_dist = d
                best_node = node
        
        if best_node == -1:
            break
        
        # Move to best_node
        d = _dist(coords, current_pos, best_node)
        time_t += d / speed
        battery -= d
        
        if best_node in instance["stations_set"]:
            time_t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            load += instance["demands"][best_node]
            ready = instance["ready"][best_node]
            service = instance["service"][best_node]
            if time_t < ready:
                time_t = ready
            time_t += service
        
        route.append(best_node)
        remaining.remove(best_node)
        current_pos = best_node
    
    # Check if we can return to depot
    if route:
        valid, _, end_time = _simulate(route, instance, coords)
        if not valid:
            # Try to trim the route until it's valid
            while route and not _simulate(route, instance, coords)[0]:
                route.pop()
            if not route:
                return [], start_time, start_battery
            # Re-simulate to get correct end time/battery
            valid, _, end_time = _simulate(route, instance, coords)
            if not valid:
                return [], start_time, start_battery
            # Update remaining
            for node in route:
                if node in remaining:
                    remaining.remove(node)
            return route, end_time, battery_max  # Battery state is complex, just return max for simplicity
    
    return [], start_time, start_battery

def _two_opt(route, instance, coords):
    """
    Perform 2-opt local search on a route.
    Returns the improved route.
    """
    n = len(route)
    if n < 3:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Check segment [i+1, j]
                # New route: route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                
                # Quick distance check before full simulation
                # d(i, route[i+1]) + d(route[j], route[j+1]) vs d(i, route[j]) + d(route[i+1], route[j+1])
                # But we also need to consider depot legs if i=0 or j=n-1
                # Actually, 2-opt on the internal sequence.
                # The distance change is:
                # Old: dist(route[i], route[i+1]) + dist(route[j], route[j+1])
                # New: dist(route[i], route[j]) + dist(route[i+1], route[j+1])
                # But wait, the reversal is route[i+1]...route[j].
                # So the connections are:
                # Old: ... -> route[i] -> route[i+1] ... route[j] -> route[j+1] -> ...
                # New: ... -> route[i] -> route[j] ... route[i+1] -> route[j+1] -> ...
                
                # We need to handle the depot for i=0 or j=n-1
                # Let's just use full simulation for correctness, but it's slow.
                # For speed, we can do a quick check.
                
                # Let's try the quick check
                node_a = route[i]
                node_b = route[i+1]
                node_c = route[j]
                node_d = route[j+1] if j+1 < n else 0
                
                # But if i=0, node_a is effectively depot? No, route[0] is the first customer.
                # The depot is implicit.
                # So the edges are:
                # (depot, route[0]), (route[0], route[1]), ..., (route[n-1], depot)
                # 2-opt on indices i, j in the customer list.
                # The affected edges are:
                # (route[i-1], route[i]) and (route[i], route[i+1]) if i>0
                # (route[j], route[j+1]) and (route[j+1], route[j+2]) if j<n-2
                # This is getting complicated. Let's just use full simulation for now, 
                # but limit the number of iterations.
                
                # Actually, a simpler 2-opt for TSP:
                # Reverse segment [i, j]
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                
                # Check validity and cost
                valid_old, cost_old, _ = _simulate(route, instance, coords)
                valid_new, cost_new, _ = _simulate(new_route, instance, coords)
                
                if valid_new and (not valid_old or cost_new < cost_old - 1e-9):
                    route = new_route
                    improved = True
                    break
            if improved:
                break
    
    return route

def _relocate(route, instance, coords):
    """
    Perform relocate local search on a route.
    """
    n = len(route)
    if n < 2:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Move route[i] to position j
                # If j > i, the index shifts
                new_route = route[:i] + route[i+1:]
                insert_pos = j if j < i else j - 1
                new_route = new_route[:insert_pos] + [route[i]] + new_route[insert_pos:]
                
                valid_old, cost_old, _ = _simulate(route, instance, coords)
                valid_new, cost_new, _ = _simulate(new_route, instance, coords)
                
                if valid_new and (not valid_old or cost_new < cost_old - 1e-9):
                    route = new_route
                    improved = True
                    break
            if improved:
                break
    
    return route

def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    instance["stations_set"] = set(stations)
    
    customers = list(range(1, n_customers + 1))
    
    # Sort customers by due time for initial ordering
    customers.sort(key=lambda v: instance["due"][v])
    
    unassigned = list(customers)
    routes = []
    
    # Constructive phase
    while unassigned:
        if time.time() > deadline:
            break
        
        route, end_time, end_battery = _constructive_route(instance, coords, unassigned)
        
        if not route:
            # No customer can be added to a new route?
            # This shouldn't happen with the baseline, but just in case
            # Force assign the next customer
            next_c = unassigned.pop(0)
            routes.append([next_c])
            continue
        
        # Remove assigned customers from unassigned
        for node in route:
            if node in unassigned:
                unassigned.remove(node)
        
        routes.append(route)
    
    # Local search phase
    for _ in range(10):  # Limit iterations
        if time.time() > deadline:
            break
        
        improved = False
        for r_idx, route in enumerate(routes):
            if time.time() > deadline:
                break
            
            # 2-opt
            new_route = _two_opt(route, instance, coords)
            if new_route != route:
                routes[r_idx] = new_route
                improved = True
            
            # Relocate
            new_route = _relocate(routes[r_idx], instance, coords)
            if new_route != routes[r_idx]:
                routes[r_idx] = new_route
                improved = True
        
        if not improved:
            break
    
    # Final validation and cleanup
    final_routes = []
    for route in routes:
        valid, cost, _ = _simulate(route, instance, coords)
        if valid:
            final_routes.append(route)
        else:
            # Try to fix by removing customers
            for node in route:
                if node in unassigned:
                    unassigned.append(node)
                elif node not in [c for r in final_routes for c in r]:
                    # Already assigned elsewhere? Shouldn't happen
                    pass
    
    # Check if all customers are assigned
    assigned = set(c for r in final_routes for c in r)
    missing = [c for c in customers if c not in assigned]
    
    if missing:
        # Assign missing customers one by one
        for c in missing:
            best_route_idx = -1
            best_cost = float('inf')
            
            for r_idx, route in enumerate(final_routes):
                # Try inserting c into route
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    valid, cost, _ = _simulate(new_route, instance, coords)
                    if valid and cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
            
            if best_route_idx != -1:
                final_routes[best_route_idx] = final_routes[best_route_idx][:best_pos] + [c] + final_routes[best_route_idx][best_pos:]
            else:
                # Create a new route
                final_routes.append([c])
    
    return final_routes
# EVOLVE-BLOCK-END
