"""EV Routing: Multi-start Randomized Construction with Aggressive Local Search and Merging."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

DEADLINE = 115.0  # Safety margin before 120s limit

def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _validate_route(route, inst, dist_mat):
    coords = inst["coords"]
    demands = inst["demands"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    cap = inst["capacity"]
    batt_max = inst["battery"]
    speed = inst["speed"]
    r_rate = inst["recharge_rate"]
    horizon = inst["horizon"]
    stations = set(inst["stations"])
    
    if not route:
        return True
        
    total_demand = sum(demands[i] for i in route)
    if total_demand > cap:
        return False
        
    t = 0.0
    batt = batt_max
    prev = 0  # depot
    
    for i in route:
        d = dist_mat[prev, i]
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return False
        
        # If station, recharge
        if i in stations:
            wait = (batt_max - batt) * r_rate
            t += wait
            batt = batt_max
        else:
            # Customer
            if t < ready[i]:
                t = ready[i]
            if t > due[i] + 1e-9:
                return False
            t += service[i]
        
        prev = i
    
    # Return to depot
    d = dist_mat[prev, 0]
    t += d / speed
    batt -= d
    if batt < -1e-9:
        return False
    if t > horizon + 1e-9:
        return False
        
    return True

def _route_distance(route, inst, dist_mat):
    if not route:
        return 0.0
    d = dist_mat[0, route[0]]
    for i in range(len(route) - 1):
        d += dist_mat[route[i], route[i+1]]
    d += dist_mat[route[-1], 0]
    return d

def _total_distance(routes, inst, dist_mat):
    return sum(_route_distance(r, inst, dist_mat) for r in routes)

def _two_opt(route, inst, dist_mat):
    """Improve route using 2-opt."""
    coords = inst["coords"]
    improved = True
    n = len(route)
    if n < 2:
        return route
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i+1 to j
                new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                if _validate_route(new_route, inst, dist_mat):
                    old_d = _route_distance(route, inst, dist_mat)
                    new_d = _route_distance(new_route, inst, dist_mat)
                    if new_d < old_d - 1e-6:
                        route = new_route
                        improved = True
    return route

def _relocate(routes, inst, dist_mat):
    """Try to relocate a customer from one route to another."""
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                cust = routes[i][j]
                # Remove from route i
                orig_route_i = routes[i][:]
                new_route_i = routes[i][:j] + routes[i][j+1:]
                # Try inserting into all other routes
                for k in range(len(routes)):
                    if k == i:
                        continue
                    for pos in range(len(routes[k]) + 1):
                        new_route_k = routes[k][:pos] + [cust] + routes[k][pos:]
                        # Check validity
                        if _validate_route(new_route_i, inst, dist_mat) and _validate_route(new_route_k, inst, dist_mat):
                            # Check capacity
                            if sum(inst["demands"][x] for x in new_route_i) <= inst["capacity"] and \
                               sum(inst["demands"][x] for x in new_route_k) <= inst["capacity"]:
                                # Calculate cost change
                                old_cost = _route_distance(orig_route_i, inst, dist_mat) + _route_distance(routes[k], inst, dist_mat)
                                new_cost = _route_distance(new_route_i, inst, dist_mat) + _route_distance(new_route_k, inst, dist_mat)
                                if new_cost < old_cost - 1e-6:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
                                    improved = True
                                    break
                    if improved:
                        break
            if improved:
                break
    return routes

def _swap(routes, inst, dist_mat):
    """Try to swap two customers between routes."""
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i >= j:
                    continue
                for a in range(len(routes[i])):
                    for b in range(len(routes[j])):
                        c1 = routes[i][a]
                        c2 = routes[j][b]
                        # Swap
                        new_route_i = routes[i][:]
                        new_route_j = routes[j][:]
                        new_route_i[a] = c2
                        new_route_j[b] = c1
                        if _validate_route(new_route_i, inst, dist_mat) and _validate_route(new_route_j, inst, dist_mat):
                            if sum(inst["demands"][x] for x in new_route_i) <= inst["capacity"] and \
                               sum(inst["demands"][x] for x in new_route_j) <= inst["capacity"]:
                                old_cost = _route_distance(routes[i], inst, dist_mat) + _route_distance(routes[j], inst, dist_mat)
                                new_cost = _route_distance(new_route_i, inst, dist_mat) + _route_distance(new_route_j, inst, dist_mat)
                                if new_cost < old_cost - 1e-6:
                                    routes[i] = new_route_i
                                    routes[j] = new_route_j
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes

def _merge_routes(routes, inst, dist_mat):
    """Try to merge two routes into one if feasible."""
    improved = True
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                # Try all combinations of merging route i and route j
                # This is expensive, so we only try a few strategies
                # Strategy 1: Append one to the other
                for r1, r2 in [(routes[i], routes[j]), (routes[j], routes[i])]:
                    # Try inserting each customer of r2 into r1
                    new_route = r1[:]
                    remaining = r2[:]
                    success = True
                    while remaining:
                        cust = remaining.pop(0)
                        inserted = False
                        for pos in range(len(new_route) + 1):
                            trial = new_route[:pos] + [cust] + new_route[pos:]
                            if _validate_route(trial, inst, dist_mat) and \
                               sum(inst["demands"][x] for x in trial) <= inst["capacity"]:
                                new_route = trial
                                inserted = True
                                break
                        if not inserted:
                            success = False
                            break
                    if success:
                        # Check if we can remove the other route
                        # We need to ensure all customers are covered
                        # Since we merged r2 into r1, we remove r2
                        new_routes = routes[:]
                        new_routes[i] = new_route
                        # Remove j (adjust index if i < j)
                        if i < j:
                            new_routes.pop(j)
                        else:
                            new_routes.pop(i)
                            new_routes[i] = new_route
                        # Validate all routes
                        valid = all(_validate_route(r, inst, dist_mat) for r in new_routes)
                        if valid:
                            old_cost = _total_distance(routes, inst, dist_mat)
                            new_cost = _total_distance(new_routes, inst, dist_mat)
                            if new_cost < old_cost - 1e-6:
                                routes = new_routes
                                improved = True
                                break
            if improved:
                break
    return routes

def plan(instance):
    start_time = time.time()
    random.seed(42)
    
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    cap = instance["capacity"]
    batt_max = instance["battery"]
    speed = instance["speed"]
    r_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]
    
    customers = list(range(1, n_cust + 1))
    
    # Pre-compute distance matrix
    n_nodes = len(coords)
    dist_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_mat[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    
    # Multi-start strategy
    best_routes = None
    best_cost = float('inf')
    
    # Run multiple randomized constructions
    num_starts = 10
    for start_idx in range(num_starts):
        if time.time() - start_time > DEADLINE:
            break
        
        # Randomize customer order
        shuffled_customers = customers[:]
        random.shuffle(shuffled_customers)
        
        routes = []
        unassigned = set(shuffled_customers)
        
        # Greedy insertion
        while unassigned:
            if time.time() - start_time > DEADLINE:
                break
            
            # Pick a customer (randomly or by some heuristic)
            # For diversity, pick randomly from unassigned
            cust = random.choice(list(unassigned))
            unassigned.remove(cust)
            
            best_global_cost = float('inf')
            best_global_route_idx = -1
            best_global_new_route = None
            
            # Try inserting into existing routes
            for idx, r in enumerate(routes):
                current_demand = sum(demands[i] for i in r)
                if current_demand + demands[cust] > cap:
                    continue
                n = len(r)
                for k in range(n + 1):
                    new_route = r[:k] + [cust] + r[k:]
                    if _validate_route(new_route, instance, dist_mat):
                        c = _route_distance(new_route, instance, dist_mat)
                        if c < best_global_cost:
                            best_global_cost = c
                            best_global_route_idx = idx
                            best_global_new_route = new_route
            
            # Check if new route is better
            new_route_single = [cust]
            if _validate_route(new_route_single, instance, dist_mat):
                cost_single = _route_distance(new_route_single, instance, dist_mat)
                if best_global_route_idx == -1 or cost_single < best_global_cost:
                    best_global_cost = cost_single
                    best_global_route_idx = -1
                    best_global_new_route = new_route_single
            
            if best_global_route_idx == -1:
                routes.append(best_global_new_route)
            else:
                routes[best_global_route_idx] = best_global_new_route
        
        # Local Search
        if time.time() - start_time < DEADLINE - 5:
            # 2-Opt on each route
            for idx in range(len(routes)):
                if time.time() - start_time > DEADLINE:
                    break
                routes[idx] = _two_opt(routes[idx], instance, dist_mat)
            
            # Relocate
            if time.time() - start_time < DEADLINE - 5:
                routes = _relocate(routes, instance, dist_mat)
            
            # Swap
            if time.time() - start_time < DEADLINE - 5:
                routes = _swap(routes, instance, dist_mat)
            
            # Merge
            if time.time() - start_time < DEADLINE - 5:
                routes = _merge_routes(routes, instance, dist_mat)
        
        # Update best
        cost = _total_distance(routes, instance, dist_mat)
        if cost < best_cost:
            best_cost = cost
            best_routes = [r[:] for r in routes]
    
    # If no valid routes found, return empty
    if best_routes is None:
        return []
    
    return best_routes

# EVOLVE-BLOCK-END
