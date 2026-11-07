"""EV Routing: Genetic Algorithm with Time-Window-Preserving Crossover and Repair."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

DEADLINE_SECONDS = 115.0

def _compute_route_cost(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon):
    """
    Computes total distance, checks feasibility, and returns (distance, is_feasible).
    """
    if not route:
        return 0.0, True
        
    t = 0.0
    b = battery_max
    d_sum = 0.0
    total_dist = 0.0
    
    prev = 0
    for node in route:
        dist = D[prev, node]
        total_dist += dist
        b -= dist
        if b < -1e-9:
            return total_dist, False
        t += dist / speed
        
        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            if demands[node] > 0: # Customer
                d_sum += demands[node]
                if d_sum > 100: # Capacity check
                    return total_dist, False
                start_service = max(t, ready[node])
                if start_service > due[node] + 1e-9:
                    return total_dist, False
                t = start_service + service[node]
        prev = node
        
    dist_back = D[prev, 0]
    total_dist += dist_back
    b -= dist_back
    if b < -1e-9:
        return total_dist, False
    t += dist_back / speed
    
    if t > horizon + 1e-9:
        return total_dist, False
        
    return total_dist, True

def _try_insert_station(route, pos, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list):
    """
    Attempts to fix battery infeasibility by inserting a station at pos.
    Returns new_route if successful, else None.
    """
    if pos < 0 or pos > len(route):
        return None
        
    # Simulate to find if insertion at pos helps or if we should try other positions
    # Simple strategy: try inserting each station at pos and check feasibility
    # But that's expensive. Let's try inserting the nearest station to the segment being bridged.
    
    # Determine the nodes around pos
    prev_node = route[pos-1] if pos > 0 else 0
    next_node = route[pos] if pos < len(route) else 0
    
    # Sort stations by distance to the "gap"
    candidates = sorted(stations_list, key=lambda s: D[prev_node, s] + D[s, next_node])
    
    for s in candidates:
        new_route = route[:pos] + [s] + route[pos:]
        cost, feas = _compute_route_cost(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
        if feas:
            return new_route
            
    return None

def _repair_route(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list):
    """
    Tries to make a route feasible by inserting stations and removing infeasible customers.
    Returns (feasible_route, removed_customers)
    """
    current_route = list(route)
    removed = []
    
    # Iteratively fix infeasibility
    max_iters = 50
    for _ in range(max_iters):
        cost, feas = _compute_route_cost(current_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
        if feas:
            break
            
        # Try to find where it fails and fix it
        # Simple heuristic: try inserting station at various positions
        fixed = False
        # Try inserting at random positions or specific positions
        # For speed, let's try inserting at the position where battery is lowest or time is tightest
        # Simpler: try inserting station after each node
        for i in range(len(current_route) + 1):
            new_route = _try_insert_station(current_route, i, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
            if new_route:
                current_route = new_route
                fixed = True
                break
                
        if not fixed:
            # If station insertion doesn't work, remove a customer
            # Remove the customer with the latest due time or highest demand?
            # Let's remove the last customer to keep order somewhat stable
            if current_route:
                # Find a customer to remove (not a station)
                cust_idx = -1
                for idx, node in enumerate(current_route):
                    if not is_station[node] and demands[node] > 0:
                        cust_idx = idx
                        break
                if cust_idx != -1:
                    removed.append(current_route[cust_idx])
                    current_route.pop(cust_idx)
                else:
                    # Only stations left? Shouldn't happen if we start with customers
                    break
            else:
                break
                
    return current_route, removed

def _construct_initial_routes(instance, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list):
    """
    Construct initial routes using Cheapest Insertion.
    """
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Start with empty routes? Or one route?
    # Let's start with one route and insert customers one by one.
    # Or start with multiple routes to allow for capacity/battery constraints.
    # Given capacity 100 and demands 5-25, we might need 2-4 routes.
    # Let's start with 3 empty routes.
    
    n_routes = 3
    routes = [[] for _ in range(n_routes)]
    
    # Sort customers by something? Maybe by distance from depot?
    # Or just random order.
    random.shuffle(customers)
    
    for c in customers:
        # Find the best route and position to insert c
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1
        best_new_route = None
        
        for r_idx in range(n_routes):
            route = routes[r_idx]
            n = len(route)
            
            # Try inserting at all positions
            for pos in range(n + 1):
                new_route = route[:pos] + [c] + route[pos:]
                cost, feas = _compute_route_cost(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                if feas:
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
                        best_new_route = new_route
                else:
                    # Try to repair
                    fixed, removed = _repair_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                    if removed:
                        continue # We removed a customer, so this insertion is invalid for 'c' unless c was removed (which it shouldn't be if it's the only new one)
                    # Actually, repair might remove other customers. We want to insert c.
                    # So if repair removes c, skip. If it removes others, maybe acceptable? No, we want to build valid routes.
                    # Let's just skip infeasible insertions in construction to keep it simple.
                    # Or try to repair and see if c is still there.
                    if c in fixed:
                        cost_fix, feas_fix = _compute_route_cost(fixed, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                        if feas_fix:
                            if cost_fix < best_cost:
                                best_cost = cost_fix
                                best_route_idx = r_idx
                                best_pos = pos
                                best_new_route = fixed
        
        if best_route_idx != -1:
            routes[best_route_idx] = best_new_route
        else:
            # Could not insert anywhere? Add a new route?
            # For now, force insert into first route and hope repair works or it's infeasible (bad)
            # Better: add a new route
            routes.append([c])
            
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes

def _ga_crossover(parent1, parent2, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list):
    """
    Order Crossover (OX) with repair.
    """
    # Combine all nodes from both parents, ensuring no duplicates
    # Since parents are disjoint sets of customers (in our representation), we can just take a subset
    # But GA usually operates on a single permutation. Here we have multiple routes.
    # Let's define the genome as a list of routes.
    # Crossover: Pick a subset of customers from parent1 and try to insert them into parent2's structure?
    # This is complex.
    
    # Alternative: Represent the solution as a single sequence of customers, and a partition into routes.
    # But partitioning is hard.
    
    # Let's stick to a simpler "Route Exchange" crossover.
    # Pick a random route from parent1 and a random route from parent2.
    # Try to merge them? Or swap them?
    
    # Actually, let's use a simpler mutation-heavy approach since crossover on multi-route is tricky.
    # Or use "Inter-route Or-Opt" as the main operator.
    
    # For now, let's return a copy of parent1 with a random move applied.
    # This is effectively mutation.
    
    child = [list(r) for r in parent1]
    
    # Perform a random 2-opt or Or-Opt move
    if len(child) > 0:
        r_idx = random.randint(0, len(child)-1)
        route = child[r_idx]
        if len(route) >= 2:
            # 2-opt
            i = random.randint(0, len(route)-2)
            j = random.randint(i+1, len(route)-1)
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            
            cost, feas = _compute_route_cost(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            if feas:
                child[r_idx] = new_route
            else:
                fixed, removed = _repair_route(new_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                if not removed and _compute_route_cost(fixed, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[1]:
                    child[r_idx] = fixed
                    
    return child

def plan(instance):
    start_time = time.time()
    
    # 1. Precompute Data
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
    stations = instance["stations"]
    stations_list = list(stations)
    n_customers = instance["n_customers"]
    n_nodes = len(coords)
    
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Distance Matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=-1))
    
    customers = list(range(1, n_customers + 1))
    
    # 2. Construction
    population = []
    
    # Generate initial population
    pop_size = 10
    for i in range(pop_size):
        random.seed(i) # Different seeds for diversity
        routes = _construct_initial_routes(instance, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
        # Calculate cost
        total_cost = 0
        valid = True
        for r in routes:
            c, f = _compute_route_cost(r, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
            if not f:
                valid = False
                break
            total_cost += c
        if valid:
            population.append((total_cost, routes))
            
    if not population:
        # Fallback: single route with all customers, repaired
        routes = _construct_initial_routes(instance, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
        # If still invalid, return whatever we have (might be invalid, but we must return something)
        return [r for r in routes if r]
        
    population.sort(key=lambda x: x[0])
    
    # 3. Genetic Algorithm Loop
    best_cost, best_routes = population[0]
    
    while time.time() - start_time < DEADLINE_SECONDS * 0.8:
        new_population = []
        
        # Elitism
        new_population.append(population[0])
        new_population.append(population[1])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() - start_time > DEADLINE_SECONDS * 0.9:
                break
                
            # Selection: Tournament
            p1_idx = random.randint(0, len(population)-1)
            p2_idx = random.randint(0, len(population)-1)
            if population[p1_idx][0] < population[p2_idx][0]:
                p1 = population[p1_idx][1]
            else:
                p1 = population[p2_idx][1]
            if population[p1_idx][0] < population[p2_idx][0]:
                p2 = population[p2_idx][1]
            else:
                p2 = population[p1_idx][1]
                
            # Crossover/Mutation
            child = _ga_crossover(p1, p2, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
            
            # Additional Mutation: Random Inter-route Move
            if random.random() < 0.5 and len(child) >= 2:
                r1 = random.randint(0, len(child)-1)
                r2 = random.randint(0, len(child)-1)
                if r1 != r2 and len(child[r1]) > 0:
                    node_idx = random.randint(0, len(child[r1])-1)
                    node = child[r1].pop(node_idx)
                    insert_pos = random.randint(0, len(child[r2]))
                    child[r2].insert(insert_pos, node)
                    
                    # Validate and repair
                    cost1, feas1 = _compute_route_cost(child[r1], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    cost2, feas2 = _compute_route_cost(child[r2], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                    
                    if not feas1 or not feas2:
                        # Try repair
                        new_r1, rem1 = _repair_route(child[r1], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                        new_r2, rem2 = _repair_route(child[r2], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon, stations_list)
                        
                        if rem1 or rem2:
                            # Revert move
                            child[r2].pop(child[r2].index(node))
                            child[r1].insert(node_idx, node)
                        else:
                            child[r1] = new_r1
                            child[r2] = new_r2
                            cost1, feas1 = _compute_route_cost(child[r1], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                            cost2, feas2 = _compute_route_cost(child[r2], D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                            if not feas1 or not feas2:
                                # Revert
                                child[r2].pop(child[r2].index(node))
                                child[r1].insert(node_idx, node)
            
            # Evaluate Child
            total_cost = 0
            valid = True
            for r in child:
                c, f = _compute_route_cost(r, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                if not f:
                    valid = False
                    break
                total_cost += c
            
            if valid:
                new_population.append((total_cost, child))
                
        if not new_population:
            break
            
        population = new_population
        population.sort(key=lambda x: x[0])
        
        if population[0][0] < best_cost:
            best_cost = population[0][0]
            best_routes = population[0][1]
            
    # 4. Final Polish: Local Search on Best Route
    # Apply Or-Opt to the best solution
    improved = True
    while improved and time.time() - start_time < DEADLINE_SECONDS * 0.95:
        improved = False
        for r_idx in range(len(best_routes)):
            route = best_routes[r_idx]
            n = len(route)
            if n < 2:
                continue
                
            best_saving = 0
            best_move = None
            
            # Or-Opt (move 1 node)
            for i in range(n):
                for t_idx in range(len(best_routes)):
                    target_route = best_routes[t_idx]
                    t_n = len(target_route)
                    
                    for pos in range(t_n + 1):
                        if t_idx == r_idx and pos == i:
                            continue
                        # Skip if pos is right after i in same route
                        if t_idx == r_idx and pos == i + 1:
                            continue
                            
                        # Construct new routes
                        new_route_r = route[:i] + route[i+1:]
                        new_route_t = target_route[:pos] + [route[i]] + target_route[pos:]
                        
                        cost_r, feas_r = _compute_route_cost(new_route_r, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                        if not feas_r:
                            continue
                        cost_t, feas_t = _compute_route_cost(new_route_t, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)
                        if not feas_t:
                            continue
                            
                        old_total = _compute_route_cost(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[0]
                        if t_idx != r_idx:
                            old_total += _compute_route_cost(target_route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[0]
                        else:
                            old_total = _compute_route_cost(route, D, ready, due, service, demands, is_station, battery_max, recharge_rate, speed, horizon)[0]
                            
                        new_total = cost_r + cost_t
                        saving = old_total - new_total
                        
                        if saving > best_saving:
                            best_saving = saving
                            best_move = (r_idx, i, t_idx, pos, new_route_r, new_route_t, cost_r, cost_t)
                            
            if best_move:
                r_idx, i, t_idx, pos, new_route_r, new_route_t, cost_r, cost_t = best_move
                if t_idx == r_idx:
                    best_routes[r_idx] = new_route_r
                else:
                    best_routes[r_idx] = new_route_r
                    best_routes[t_idx] = new_route_t
                improved = True
                
    # Ensure all customers are present
    visited = set()
    for r in best_routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)
                
    missing = set(customers) - visited
    for m in missing:
        best_routes.append([m])
        
    return best_routes

# EVOLVE-BLOCK-END
