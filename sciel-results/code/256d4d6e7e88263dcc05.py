"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

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
    dist_list = dist.tolist()  # Faster for scalar access

    station_set = set(stations)

    # Fast route simulation: returns (total_dist, valid) or (inf, False)
    # route is a list of node ids (customers and stations)
    def simulate_route(route):
        """Simulate route, return (total_distance, is_valid)."""
        t = 0.0
        bat = float(battery_cap)
        load = 0
        total_d = 0.0
        prev = 0
        for node in route:
            d = dist_list[prev][node]
            if bat < d - 1e-9:
                return (float('inf'), False)
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity:
                return (float('inf'), False)
            if 1 <= node <= n_customers:
                if t < ready[node] - 1e-9:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return (float('inf'), False)
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                t += max(0, deficit * recharge_rate)
                bat = battery_cap
            prev = node
            total_d += d
        d = dist_list[prev][0]
        if bat < d - 1e-9:
            return (float('inf'), False)
        t += d / speed
        if t > horizon + 1e-9:
            return (float('inf'), False)
        total_d += d
        return (total_d, True)

    # Compute total distance of a route
    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist_list[prev][node]
            prev = node
        total += dist_list[prev][0]
        return total

    # Validate a route thoroughly
    def validate_route(route):
        if not route:
            return True
        _, valid = simulate_route(route)
        return valid

    # Build a route given a sequence of customer visits (with stations inserted optimally)
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
        bat = float(battery_cap)
        load = 0
        prev = 0
        
        for cust in cust_seq:
            # Try to go directly from prev to cust
            d_direct = dist_list[prev][cust]
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
                # Need to visit a station first - choose best station
                best_st = None
                best_t = float('inf')
                for st in stations:
                    d_to_st = dist_list[prev][st]
                    if bat < d_to_st:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    # Recharge
                    deficit = battery_cap - bat_at_st
                    recharge_time = deficit * recharge_rate
                    t_st += recharge_time
                    bat_st = battery_cap
                    d_st_to_cust = dist_list[st][cust]
                    if bat_st < d_st_to_cust:
                        continue
                    t_cust = t_st + d_st_to_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust]:
                        continue
                    if t_cust < best_t:
                        best_t = t_cust
                        best_st = st
                if best_st is None:
                    return None
                # Use best station
                d_to_st = dist_list[prev][best_st]
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_cust = dist_list[best_st][cust]
                t_cust = t_st + d_st_to_cust / speed
                if t_cust < ready[cust]:
                    t_cust = ready[cust]
                t = t_cust + service[cust]
                bat = bat_st - d_st_to_cust
                load += demands[cust]
                route.append(best_st)
                route.append(cust)
                prev = cust
        
        # Check return to depot
        d_return = dist_list[prev][0]
        if bat < d_return:
            # Try to go through a station
            best_st = None
            best_t = float('inf')
            for st in stations:
                d_to_st = dist_list[prev][st]
                if bat < d_to_st:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_depot = dist_list[st][0]
                if bat_st < d_st_to_depot:
                    continue
                t_depot = t_st + d_st_to_depot / speed
                if t_depot > horizon:
                    continue
                if t_depot < best_t:
                    best_t = t_depot
                    best_st = st
            if best_st is None:
                return None
            route.append(best_st)
        else:
            t_return = t + d_return / speed
            if t_return > horizon:
                return None
        
        return route

    # Clarke-Wright savings based construction
    def clarke_wright_savings():
        """Build initial routes using Clarke-Wright savings heuristic."""
        customers = list(range(1, n_customers + 1))
        
        # Initial: each customer in its own route
        routes = [[c] for c in customers]
        route_of = {c: i for i, c in enumerate(customers)}
        
        # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(1, n_customers + 1):
            for j in range(i + 1, n_customers + 1):
                s = dist_list[0][i] + dist_list[0][j] - dist_list[i][j]
                if s > 0:
                    savings.append((-s, i, j))  # Negative for min-heap (max savings first)
        savings.sort()
        
        # Merge routes based on savings
        for neg_s, i, j in savings:
            if time.time() > deadline - 20.0:
                break
            ri = route_of.get(i)
            rj = route_of.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            
            route_i = routes[ri]
            route_j = routes[rj]
            
            # Check if i is at end of route_i or j is at end of route_j (or vice versa)
            # Try to merge: i must be last in route_i, j must be first in route_j
            # Or j must be last in route_j, i must be first in route_i
            
            merged = None
            # Case 1: i at end of route_i, j at start of route_j
            if route_i[-1] == i and route_j[0] == j:
                new_route = route_i + route_j
                if sum(demands[c] for c in new_route) <= capacity:
                    test_route = build_route_from_sequence(new_route)
                    if test_route is not None and validate_route(test_route):
                        merged = new_route
            # Case 2: j at end of route_j, i at start of route_i
            if merged is None and route_j[-1] == j and route_i[0] == i:
                new_route = route_j + route_i
                if sum(demands[c] for c in new_route) <= capacity:
                    test_route = build_route_from_sequence(new_route)
                    if test_route is not None and validate_route(test_route):
                        merged = new_route
            # Case 3: i at end of route_i, j at end of route_j (reverse route_j)
            if merged is None and route_i[-1] == i and route_j[-1] == j:
                new_route = route_i + route_j[::-1]
                if sum(demands[c] for c in new_route) <= capacity:
                    test_route = build_route_from_sequence(new_route)
                    if test_route is not None and validate_route(test_route):
                        merged = new_route
            # Case 4: j at start of route_j, i at start of route_i (reverse route_i)
            if merged is None and route_j[0] == j and route_i[0] == i:
                new_route = route_j + route_i[::-1]
                if sum(demands[c] for c in new_route) <= capacity:
                    test_route = build_route_from_sequence(new_route)
                    if test_route is not None and validate_route(test_route):
                        merged = new_route
            
            if merged is not None:
                # Remove route_j, merge into route_i
                routes[ri] = merged
                routes[rj] = []
                for c in merged:
                    route_of[c] = ri
                # Clean up empty routes
                routes = [r for r in routes if r]
                route_of = {c: routes.index(r) for r in routes for c in r}
        
        return [build_route_from_sequence(r) for r in routes if r]

    # Greedy construction with 2-opt local search
    customers = list(range(1, n_customers + 1))
    
    # Sort by due time for initial ordering
    customers.sort(key=lambda v: due[v])
    
    # Build routes greedily
    routes = []
    remaining = set(customers)
    
    while remaining and time.time() < deadline - 10.0:
        # Find the best next route using nearest neighbor with time window feasibility
        best_route = None
        best_dist = float('inf')
        best_used = set()
        
        # Try starting from each remaining customer (limited for speed)
        start_candidates = list(remaining)
        start_candidates.sort(key=lambda v: due[v])
        
        for start_cust in start_candidates[:8]:  # Limit for speed
            # Build a route starting with this customer using greedy insertion
            route_seq = [start_cust]
            used = {start_cust}
            current_load = demands[start_cust]
            
            # Greedily add the nearest feasible customer
            while time.time() < deadline - 10.0:
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

    # Try Clarke-Wright and keep better solution
    if time.time() < deadline - 30.0:
        cw_routes = clarke_wright_savings()
        if cw_routes:
            cw_valid = [r for r in cw_routes if r is not None and validate_route(r)]
            cw_dist = sum(route_distance(r) for r in cw_valid)
            current_dist = sum(route_distance(r) for r in routes)
            
            # Check coverage
            cw_covered = set()
            for r in cw_valid:
                for node in r:
                    if 1 <= node <= n_customers:
                        cw_covered.add(node)
            
            if cw_dist < current_dist and len(cw_covered) == n_customers:
                routes = cw_valid

    # Local search: 2-opt within routes and move customers between routes
    if time.time() < deadline - 15.0:
        # Precompute route distances for quick comparison
        route_dists = [route_distance(r) for r in routes]
        
        # Try to improve by moving customers between routes (relocate)
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 10.0:
            improved = False
            iteration += 1
            
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline - 10.0:
                        break
                    cust = routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue
                    
                    # Try moving to another route
                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 10.0:
                            continue
                        
                        # Quick feasibility check: can route k accommodate this customer?
                        current_load_k = sum(demands[c] for c in routes[k] if 1 <= c <= n_customers)
                        if current_load_k + demands[cust] > capacity:
                            continue
                        
                        # Try inserting at each position in route k
                        for pos in range(len(routes[k]) + 1):
                            if time.time() > deadline - 10.0:
                                break
                            
                            # Build new sequences
                            new_route_i = routes[i][:j] + routes[i][j+1:]
                            new_route_k = routes[k][:pos] + [cust] + routes[k][pos:]
                            
                            # Quick distance estimate before full validation
                            if not new_route_i:
                                dist_i = 0.0
                            else:
                                prev = 0
                                dist_i = 0.0
                                for node in new_route_i:
                                    dist_i += dist_list[prev][node]
                                    prev = node
                                dist_i += dist_list[prev][0]
                            
                            if not new_route_k:
                                dist_k = 0.0
                            else:
                                prev = 0
                                dist_k = 0.0
                                for node in new_route_k:
                                    dist_k += dist_list[prev][node]
                                    prev = node
                                dist_k += dist_list[prev][0]
                            
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
        if time.time() < deadline - 10.0:
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
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
                    if time.time() > deadline - 10.0:
                        break
                    for b in range(a+1, len(best_seq)):
                        if time.time() > deadline - 10.0:
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

    # Or-opt: move 1-3 consecutive customers within or between routes
    if time.time() < deadline - 10.0:
        route_dists = [route_distance(r) for r in routes]
        improved = True
        max_oropt_iter = 30
        oropt_iter = 0
        while improved and oropt_iter < max_oropt_iter and time.time() < deadline - 10.0:
            improved = False
            oropt_iter += 1
            
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline - 10.0:
                        break
                    cust_seq_i = [c for c in routes[i] if 1 <= c <= n_customers]
                    for start in range(max(0, len(cust_seq_i) - seg_len + 1)):
                        if time.time() > deadline - 10.0:
                            break
                        segment = cust_seq_i[start:start + seg_len]
                        
                        # Remove segment from route i
                        remaining_seq_i = cust_seq_i[:start] + cust_seq_i[start + seg_len:]
                        
                        # Try inserting into route i at different positions
                        for pos in range(len(remaining_seq_i) + 1):
                            if pos == start:
                                continue
                            new_seq = remaining_seq_i[:pos] + segment + remaining_seq_i[pos:]
                            new_route = build_route_from_sequence(new_seq)
                            if new_route is not None and validate_route(new_route):
                                d = route_distance(new_route)
                                if d < route_dists[i] - 1e-6:
                                    routes[i] = new_route
                                    route_dists[i] = d
                                    improved = True
                                    break
                        if improved:
                            break
                        
                        # Try inserting into other routes
                        for k in range(len(routes)):
                            if k == i or time.time() > deadline - 10.0:
                                continue
                            cust_seq_k = [c for c in routes[k] if 1 <= c <= n_customers]
                            current_load_k = sum(demands[c] for c in cust_seq_k)
                            seg_load = sum(demands[c] for c in segment)
                            if current_load_k + seg_load > capacity:
                                continue
                            for pos in range(len(cust_seq_k) + 1):
                                if time.time() > deadline - 10.0:
                                    break
                                new_seq_k = cust_seq_k[:pos] + segment + cust_seq_k[pos:]
                                new_route_k = build_route_from_sequence(new_seq_k)
                                new_route_i = build_route_from_sequence(remaining_seq_i)
                                if new_route_k is None or new_route_i is None:
                                    continue
                                if not validate_route(new_route_k) or not validate_route(new_route_i):
                                    continue
                                new_d = route_distance(new_route_k) + route_distance(new_route_i)
                                if new_d < route_dists[i] + route_dists[k] - 1e-6:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
                                    route_dists[i] = route_distance(new_route_i)
                                    route_dists[k] = route_distance(new_route_k)
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

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
