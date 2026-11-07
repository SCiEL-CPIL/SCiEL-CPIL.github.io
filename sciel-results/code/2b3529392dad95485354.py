"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances using NumPy vectorization
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    # Build a route given a sequence of customer visits (with stations inserted optimally)
    # For a given sequence of customers, find the best station insertions
    def build_route_from_sequence(cust_seq):
        """
        Given a sequence of customers, build a valid route by inserting stations as needed.
        Returns the route (list of node ids) or None if infeasible.
        """
        if not cust_seq:
            return []
        
        # Quick capacity check
        total_load = sum(demands[c] for c in cust_seq)
        if total_load > capacity:
            return None
        
        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        
        for cust in cust_seq:
            # Try to go directly from prev to cust
            d_direct = dist[prev][cust]
            if bat >= d_direct:
                # Can go directly
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust]:
                    return None  # Time window violation
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                route.append(cust)
                prev = cust
            else:
                # Need to visit a station first
                found = False
                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    # Recharge
                    deficit = battery_cap - bat_at_st
                    recharge_time = deficit * recharge_rate
                    t_st += recharge_time
                    bat_st = battery_cap
                    d_st_to_cust = dist[st][cust]
                    if bat_st < d_st_to_cust:
                        continue
                    t_cust = t_st + d_st_to_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust]:
                        continue
                    t = t_cust + service[cust]
                    bat = bat_st - d_st_to_cust
                    load += demands[cust]
                    route.append(st)
                    route.append(cust)
                    prev = cust
                    found = True
                    break
                if not found:
                    return None
        
        # Check return to depot
        d_return = dist[prev][0]
        if bat < d_return:
            # Try to go through a station
            found = False
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_depot = dist[st][0]
                if bat_st < d_st_to_depot:
                    continue
                t_depot = t_st + d_st_to_depot / speed
                if t_depot > horizon:
                    continue
                route.append(st)
                found = True
                break
            if not found:
                return None
        else:
            t_return = t + d_return / speed
            if t_return > horizon:
                return None
        
        return route

    # Compute total distance of a route
    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    # Validate a route thoroughly
    def validate_route(route):
        if not route:
            return True
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        for node in route:
            d = dist[prev][node]
            if bat < d - 1e-9:
                return False
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return False
            if 1 <= node <= n_customers:
                if t < ready[node] - 1e-9:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False
                t += service[node]
            elif node in stations:
                deficit = battery_cap - bat
                t += max(0, deficit * recharge_rate)
                bat = battery_cap
            prev = node
        d = dist[prev][0]
        if bat < d - 1e-9:
            return False
        t += d / speed
        if t > horizon + 1e-9:
            return False
        return True

    # Greedy construction: build routes by adding customers one at a time
    # Sort customers by due time for initial ordering
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])

    # Try to build a single route with all customers (likely infeasible due to capacity)
    # Then split into multiple routes using a greedy approach

    # First, check if all customers can fit in one route capacity-wise
    total_demand = sum(demands[c] for c in customers)
    if total_demand <= capacity:
        # Try to build a single route
        # Use nearest neighbor with time window feasibility
        route = build_route_from_sequence(customers)
        if route is not None and validate_route(route):
            return [route]

    # Build routes greedily
    routes = []
    remaining = set(customers)
    
    # Try different starting customer strategies
    while remaining and time.time() < deadline - 5.0:
        # Try to find the best next route
        best_route = None
        best_dist = float('inf')
        
        # Try starting from each remaining customer (limited for speed)
        start_candidates = list(remaining)
        # Sort by due time
        start_candidates.sort(key=lambda v: due[v])
        
        for start_cust in start_candidates[:5]:  # Limit for speed
            # Build a route starting with this customer using greedy insertion
            route_seq = [start_cust]
            used = {start_cust}
            current_load = demands[start_cust]
            
            # Greedily add the nearest feasible customer
            while True:
                best_next = None
                best_metric = float('inf')
                
                for c in remaining - used:
                    if current_load + demands[c] > capacity:
                        continue
                    # Check if we can add c to the current sequence
                    new_seq = route_seq + [c]
                    test_route = build_route_from_sequence(new_seq)
                    if test_route is not None:
                        d = route_distance(test_route)
                        if d < best_metric:
                            best_metric = d
                            best_next = c
                
                if best_next is None:
                    break
                route_seq.append(best_next)
                used.add(best_next)
                current_load += demands[best_next]
            
            # Finalize the route
            final_route = build_route_from_sequence(route_seq)
            if final_route is not None and validate_route(final_route):
                d = route_distance(final_route)
                if d < best_dist:
                    best_dist = d
                    best_route = final_route
                    best_used = set(c for c in route_seq if 1 <= c <= n_customers)
        
        if best_route is None:
            # Fallback: single customer route
            for c in remaining:
                single_route = build_route_from_sequence([c])
                if single_route is not None and validate_route(single_route):
                    best_route = single_route
                    best_used = {c}
                    break
            if best_route is None:
                # Should not happen, but add as-is
                best_route = [list(remaining)[0]]
                best_used = {list(remaining)[0]}
        
        routes.append(best_route)
        remaining -= best_used
    
    # Add any remaining customers as single routes
    for c in remaining:
        single_route = build_route_from_sequence([c])
        if single_route is not None:
            routes.append(single_route)
        else:
            routes.append([c])

    # Local search: 2-opt within routes and move customers between routes
    if time.time() < deadline - 10.0:
        # Precompute route distances for quick comparison
        route_dists = [route_distance(r) for r in routes]
        
        # Try to improve by moving customers between routes (relocate)
        improved = True
        max_iterations = 30
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 5.0:
            improved = False
            iteration += 1
            
            for i in range(len(routes)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline - 5.0:
                        break
                    cust = routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue
                    
                    # Quick check: if removing this customer makes route infeasible, skip
                    # (we'll still try, but this helps prioritize)
                    
                    # Try moving to another route
                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 5.0:
                            continue
                        
                        # Quick feasibility check: can route k accommodate this customer?
                        current_load_k = sum(demands[c] for c in routes[k] if 1 <= c <= n_customers)
                        if current_load_k + demands[cust] > capacity:
                            continue
                        
                        # Try inserting at each position in route k
                        for pos in range(len(routes[k]) + 1):
                            if time.time() > deadline - 5.0:
                                break
                            
                            # Build new sequences
                            new_route_i = routes[i][:j] + routes[i][j+1:]
                            new_route_k = routes[k][:pos] + [cust] + routes[k][pos:]
                            
                            # Quick distance estimate before full validation
                            # This is an approximation to avoid expensive calls
                            if not new_route_i:
                                dist_i = 0.0
                            else:
                                prev = 0
                                dist_i = 0.0
                                for node in new_route_i:
                                    dist_i += dist[prev][node]
                                    prev = node
                                dist_i += dist[prev][0]
                            
                            if not new_route_k:
                                dist_k = 0.0
                            else:
                                prev = 0
                                dist_k = 0.0
                                for node in new_route_k:
                                    dist_k += dist[prev][node]
                                    prev = node
                                dist_k += dist[prev][0]
                            
                            # If estimated distance is not better, skip full validation
                            if dist_i + dist_k >= route_dists[i] + route_dists[k] - 0.5:
                                continue
                            
                            # Full validation
                            r_i = build_route_from_sequence(new_route_i)
                            r_k = build_route_from_sequence(new_route_k)
                            
                            if r_i is None or r_k is None:
                                continue
                            if not validate_route(r_i) or not validate_route(r_k):
                                continue
                            
                            new_dist = route_distance(r_i) + route_distance(r_k)
                            
                            if new_dist < route_dists[i] + route_dists[k] - 1e-6:
                                routes[i] = r_i
                                routes[k] = r_k
                                route_dists[i] = route_distance(r_i)
                                route_dists[k] = route_distance(r_k)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Try 2-opt within each route
        if time.time() < deadline - 5.0:
            for i in range(len(routes)):
                if time.time() > deadline - 5.0:
                    break
                # Extract customer sequence
                cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
                if len(cust_seq) < 2:
                    continue
                
                current_route = build_route_from_sequence(cust_seq)
                if current_route is None:
                    continue
                best_dist = route_distance(current_route)
                best_seq = cust_seq[:]
                
                # Try reversing segments
                for a in range(len(best_seq)):
                    if time.time() > deadline - 5.0:
                        break
                    for b in range(a+1, len(best_seq)):
                        if time.time() > deadline - 5.0:
                            break
                        new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                        new_route = build_route_from_sequence(new_seq)
                        if new_route is not None and validate_route(new_route):
                            d = route_distance(new_route)
                            if d < best_dist - 1e-6:
                                best_dist = d
                                best_seq = new_seq[:]
                
                if best_seq != cust_seq:
                    new_route = build_route_from_sequence(best_seq)
                    if new_route is not None and validate_route(new_route):
                        routes[i] = new_route

    # Final validation
    valid_routes = []
    covered = set()
    for r in routes:
        if validate_route(r):
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    # Add any missing customers
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            single_route = build_route_from_sequence([cust])
            if single_route is not None:
                valid_routes.append(single_route)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
# EVOLVE-BLOCK-END
