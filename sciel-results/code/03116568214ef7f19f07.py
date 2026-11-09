"""Greedy Insertion Construction with Or-Opt and Relocate Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import random

def _euclidean(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def _check_route(coords, demands, ready, due, service, route, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """Check if a route is feasible. Returns (feasible, total_dist)."""
    if not route:
        return True, 0.0
    
    total_demand = sum(demands[n] for n in route)
    if total_demand > capacity:
        return False, 0.0
    
    time = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0
    
    for node in route:
        d = _euclidean(coords[prev], coords[node])
        total_dist += d
        time += d / speed
        battery -= d
        
        if battery < -1e-9:
            return False, 0.0
        
        if node in stations:
            energy_needed = battery_max - battery
            if energy_needed > 1e-9:
                time += energy_needed * recharge_rate
            battery = battery_max
        else:
            if time < ready[node]:
                time = ready[node]
            if time > due[node] + 1e-9:
                return False, 0.0
            time += service[node]
        
        prev = node
    
    d = _euclidean(coords[prev], coords[0])
    total_dist += d
    time += d / speed
    battery -= d
    
    if battery < -1e-9:
        return False, 0.0
    if time > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist

def _route_distance(coords, route):
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for node in route:
        total += _euclidean(coords[prev], coords[node])
        prev = node
    total += _euclidean(coords[prev], coords[0])
    return total

def _construct_greedy_insertion(coords, demands, ready, due, service, customers, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """
    Construct routes by iteratively inserting the 'best' customer into the 'best' position.
    We maintain a list of unserved customers. In each step, we try to insert each unserved customer
    into every possible position in every existing route (or start a new route) and pick the one
    that minimizes the increase in total distance, subject to feasibility.
    """
    unserved = set(customers)
    routes = []
    
    # Pre-compute distances? Might be too large, let's compute on fly for simplicity first.
    # N=25 is small.
    
    while unserved:
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_customer = None
        best_new_routes = None
        
        # Try inserting each unserved customer
        for c in list(unserved):
            # Option 1: Start a new route
            new_route = [c]
            feasible, dist = _check_route(coords, demands, ready, due, service, new_route,
                                          capacity, battery_max, speed, recharge_rate, horizon, stations)
            if feasible:
                # Cost is just the distance of this new route
                # We want to minimize total distance, so adding a new route adds 'dist' to total.
                # However, starting a new route is usually worse than inserting if possible.
                # Let's compare insertion costs.
                pass
            
            # Option 2: Insert into existing routes
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                # Try all positions 0 to len(route)
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    feasible, dist = _check_route(coords, demands, ready, due, service, new_route,
                                                  capacity, battery_max, speed, recharge_rate, horizon, stations)
                    if feasible:
                        # Calculate delta cost
                        # Old cost: _route_distance(coords, route)
                        # New cost: dist
                        # We want to minimize the increase.
                        # To avoid recalculating old distance every time, we could cache it, 
                        # but N is small.
                        old_dist = _route_distance(coords, route)
                        delta = dist - old_dist
                        
                        if delta < best_cost:
                            best_cost = delta
                            best_route_idx = r_idx
                            best_pos = pos
                            best_customer = c
                            # We don't store new_routes yet, we reconstruct after selection
                            
            # If no insertion into existing routes was found/feasible, start a new route
            if best_customer is None or (best_customer != c and best_cost > 0 and best_route_idx == -1):
                 # Actually, the logic above sets best_customer only if an insertion is found.
                 # If no insertion was found for ANY customer, we need a new route.
                 # Let's handle the case where we couldn't insert ANY customer into existing routes.
                 pass
        
        # If no insertion was found for any customer into any route, we must start a new route for the first available customer
        if best_customer is None:
            # Pick the first unserved customer (or closest to depot)
            # Let's pick the one closest to depot to start the route
            c = min(unserved, key=lambda x: _euclidean(coords[0], coords[x]))
            routes.append([c])
            unserved.remove(c)
        else:
            # Perform the insertion
            route = routes[best_route_idx]
            new_route = route[:best_pos] + [best_customer] + route[best_pos:]
            routes[best_route_idx] = new_route
            unserved.remove(best_customer)
            
    return routes

def _local_search(coords, demands, ready, due, service, routes, capacity, battery_max, speed, recharge_rate, horizon, stations):
    """
    Perform local search using Or-Opt (move segments of 1, 2, 3) and Relocate.
    """
    current_routes = [r[:] for r in routes]
    current_dist = sum(_route_distance(coords, r) for r in current_routes)
    
    improved = True
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        # 1. Or-Opt: Move segment of length L (1, 2, 3) from pos i to pos j in same route
        #    Or move to another route.
        for L in [1, 2, 3]:
            for r_idx in range(len(current_routes)):
                route = current_routes[r_idx]
                if len(route) < L:
                    continue
                
                for i in range(len(route) - L + 1):
                    segment = route[i:i+L]
                    # Remove segment
                    base_route = route[:i] + route[i+L:]
                    
                    # Try inserting segment back into same route at different positions
                    # Positions are relative to base_route length
                    for j in range(len(base_route) + 1):
                        if j == i or (j > i and j < i + L):
                            continue # Same position or overlapping
                            
                        new_route = base_route[:j] + segment + base_route[j:]
                        
                        # Check if this is just a reversal or no-op? 
                        # Or-Opt usually checks if the segment is moved to a different place.
                        # If j == i, it's the same.
                        
                        feasible, new_dist = _check_route(coords, demands, ready, due, service, new_route,
                                                          capacity, battery_max, speed, recharge_rate, horizon, stations)
                        if feasible:
                            old_dist = _route_distance(coords, route)
                            if new_dist < old_dist - 1e-9:
                                current_routes[r_idx] = new_route
                                current_dist += new_dist - old_dist
                                improved = True
                                break # Restart loop for this L? Or just continue? 
                                # Breaking out of j loop is good.
                    if improved:
                        break # Restart i loop? 
                if improved:
                    break # Restart r_idx loop?
            
            if improved:
                break # Restart L loop?
        
        if improved:
            continue
            
        # 2. Relocate: Move single customer from route A to route B
        for r1 in range(len(current_routes)):
            for r2 in range(len(current_routes)):
                if r1 == r2:
                    continue
                
                route1 = current_routes[r1]
                route2 = current_routes[r2]
                
                for pos1 in range(len(route1)):
                    node = route1[pos1]
                    source = route1[:pos1] + route1[pos1+1:]
                    
                    for pos2 in range(len(route2) + 1):
                        target = route2[:pos2] + [node] + route2[pos2:]
                        
                        feas1, dist1 = _check_route(coords, demands, ready, due, service, source,
                                                    capacity, battery_max, speed, recharge_rate, horizon, stations)
                        feas2, dist2 = _check_route(coords, demands, ready, due, service, target,
                                                    capacity, battery_max, speed, recharge_rate, horizon, stations)
                        
                        if feas1 and feas2:
                            old_dist = _route_distance(coords, route1) + _route_distance(coords, route2)
                            new_dist = dist1 + dist2
                            if new_dist < old_dist - 1e-9:
                                current_routes[r1] = source
                                current_routes[r2] = target
                                current_dist += new_dist - old_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
                
        # Remove empty routes
        current_routes = [r for r in current_routes if r]
        
    return current_routes

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    # Construction: Greedy Insertion
    routes = _construct_greedy_insertion(coords, demands, ready, due, service, customers,
                                         capacity, battery_max, speed, recharge_rate, horizon, stations)
    
    # Ensure all customers are served (safety check)
    served = set()
    for r in routes:
        for n in r:
            if n not in stations:
                served.add(n)
                
    unserved = [c for c in customers if c not in served]
    if unserved:
        # Fallback: add them to any feasible route or new route
        for c in unserved:
            inserted = False
            for r_idx in range(len(routes)):
                for pos in range(len(routes[r_idx]) + 1):
                    new_route = routes[r_idx][:pos] + [c] + routes[r_idx][pos:]
                    feasible, _ = _check_route(coords, demands, ready, due, service, new_route,
                                               capacity, battery_max, speed, recharge_rate, horizon, stations)
                    if feasible:
                        routes[r_idx] = new_route
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                routes.append([c])
                
    # Local Search
    routes = _local_search(coords, demands, ready, due, service, routes,
                           capacity, battery_max, speed, recharge_rate, horizon, stations)
    
    return routes
# EVOLVE-BLOCK-END
