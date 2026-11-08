# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _expand_route(customer_route, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    """
    Takes a list of customer IDs and returns a list of node IDs (including stations) 
    that forms a feasible route, or None if infeasible.
    Also returns the total distance for scoring.
    """
    if not customer_route:
        return [], 0.0
        
    route = []
    current_node = 0
    bat = battery_cap
    total_time = 0.0
    total_dist = 0.0
    
    for c in customer_route:
        # Travel to c, inserting stations as needed
        while True:
            d = dist[current_node, c]
            if bat >= d:
                break
            # Need to charge
            best_s = None
            best_d_s = float('inf')
            for s in stations:
                d_to_s = dist[current_node, s]
                if bat >= d_to_s:
                    if dist[s, c] <= battery_cap:
                        if d_to_s < best_d_s:
                            best_d_s = d_to_s
                            best_s = s
            
            if best_s is None:
                return None, 0.0
            
            d_to_s = dist[current_node, best_s]
            total_time += d_to_s / speed
            bat -= d_to_s
            total_dist += d_to_s
            total_time += (battery_cap - bat) * recharge_rate
            bat = battery_cap
            route.append(best_s)
            current_node = best_s
            
        d = dist[current_node, c]
        total_time += d / speed
        bat -= d
        total_dist += d
        arr_time = total_time
        start_service = max(arr_time, ready[c])
        if start_service > due[c]:
            return None, 0.0
        total_time = start_service + service[c]
        route.append(c)
        current_node = c
        
    # Return to depot
    while True:
        d = dist[current_node, 0]
        if bat >= d:
            break
        best_s = None
        best_d_s = float('inf')
        for s in stations:
            d_to_s = dist[current_node, s]
            if bat >= d_to_s:
                if dist[s, 0] <= battery_cap:
                    if d_to_s < best_d_s:
                        best_d_s = d_to_s
                        best_s = s
        
        if best_s is None:
            return None, 0.0
            
        d_to_s = dist[current_node, best_s]
        total_time += d_to_s / speed
        bat -= d_to_s
        total_dist += d_to_s
        total_time += (battery_cap - bat) * recharge_rate
        bat = battery_cap
        route.append(best_s)
        current_node = best_s
        
    d = dist[current_node, 0]
    total_time += d / speed
    bat -= d
    total_dist += d
    
    if total_time > horizon:
        return None, 0.0
        
    return route, total_dist

def _solve_instance(instance, deadline):
    coords = np.array(instance["coords"])
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances vectorized
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    
    customers = list(range(1, n_customers + 1))
    station_set = set(stations)
    
    def extract_customers(route):
        return [node for node in route if node not in station_set and node != 0]
    
    def try_expand(customer_seq):
        return _expand_route(customer_seq, dist, ready, due, service, battery, speed, 
                             recharge_rate, horizon, stations)
    
    # Phase 1: Construct initial routes using nearest-neighbor with time window awareness
    routes = []
    unserved = set(customers)
    
    while unserved and time.time() < deadline - 20:
        # Start new route from depot
        current_customers = []
        current_demand = 0
        current_pos = 0
        current_bat = battery
        current_time = 0.0
        
        while unserved and time.time() < deadline - 20:
            best_customer = None
            best_score = float('inf')
            
            for c in unserved:
                if current_demand + demands[c] > capacity:
                    continue
                
                # Estimate feasibility and distance
                d_to_c = dist[current_pos, c]
                
                # Check if we can reach c with current battery (possibly via station)
                can_reach = False
                extra_dist = 0.0
                if current_bat >= d_to_c:
                    can_reach = True
                else:
                    for s in stations:
                        d_to_s = dist[current_pos, s]
                        if current_bat >= d_to_s and dist[s, c] <= battery:
                            can_reach = True
                            extra_dist = d_to_s + dist[s, c] - d_to_c
                            break
                
                if not can_reach:
                    continue
                
                # Estimate arrival time
                travel_time = (d_to_c + extra_dist) / speed
                arr_time = current_time + travel_time
                start_service = max(arr_time, ready[c])
                
                if start_service > due[c]:
                    continue
                
                # Score: prefer closer customers with earlier due times
                score = d_to_c + extra_dist + 0.1 * due[c]
                
                if score < best_score:
                    best_score = score
                    best_customer = c
            
            if best_customer is None:
                break
            
            # Add customer to route
            current_customers.append(best_customer)
            current_demand += demands[best_customer]
            unserved.discard(best_customer)
            
            # Update state
            c = best_customer
            d_to_c = dist[current_pos, c]
            
            # Travel to c
            if current_bat >= d_to_c:
                current_time += d_to_c / speed
                current_bat -= d_to_c
            else:
                # Go to station first
                best_s = None
                best_d_s = float('inf')
                for s in stations:
                    d_to_s = dist[current_pos, s]
                    if current_bat >= d_to_s and dist[s, c] <= battery:
                        if d_to_s < best_d_s:
                            best_d_s = d_to_s
                            best_s = s
                if best_s is not None:
                    d_to_s = dist[current_pos, best_s]
                    current_time += d_to_s / speed
                    current_bat -= d_to_s
                    current_time += (battery - current_bat) * recharge_rate
                    current_bat = battery
                    d_to_c = dist[best_s, c]
                    current_time += d_to_c / speed
                    current_bat -= d_to_c
            
            # Service
            start_service = max(current_time, ready[c])
            current_time = start_service + service[c]
            current_pos = c
        
        if current_customers:
            expanded, _ = try_expand(current_customers)
            if expanded is not None:
                routes.append(expanded)
            else:
                # Fallback: serve each customer individually
                for c in current_customers:
                    exp_single, _ = try_expand([c])
                    if exp_single is not None:
                        routes.append(exp_single)
                    else:
                        routes.append([c])
    
    # Handle remaining unserved customers
    if unserved and time.time() < deadline - 10:
        for c in sorted(unserved, key=lambda x: due[x]):
            expanded, _ = try_expand([c])
            if expanded is not None:
                routes.append(expanded)
            else:
                routes.append([c])
    
    # Phase 2: Local search optimization
    def two_opt_and_oropt(customer_seq):
        """Apply 2-opt and or-opt moves to improve customer sequence."""
        if len(customer_seq) < 2:
            return customer_seq
        
        best_seq = list(customer_seq)
        best_expanded, best_dist = try_expand(best_seq)
        if best_expanded is None:
            return customer_seq
        
        improved = True
        max_iterations = 15
        iterations = 0
        while improved and iterations < max_iterations and time.time() < deadline - 5:
            improved = False
            iterations += 1
            
            # 2-opt moves
            for i in range(len(best_seq) - 1):
                for j in range(i + 1, len(best_seq)):
                    new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                    new_expanded, new_dist = try_expand(new_seq)
                    if new_expanded is not None and new_dist < best_dist - 1e-9:
                        best_seq = new_seq
                        best_expanded = new_expanded
                        best_dist = new_dist
                        improved = True
            
            # Or-opt: move segments of length 1-3
            for seg_len in [1, 2, 3]:
                if len(best_seq) < seg_len + 1:
                    continue
                for i in range(len(best_seq) - seg_len):
                    segment = best_seq[i:i+seg_len]
                    remaining = best_seq[:i] + best_seq[i+seg_len:]
                    for j in range(len(remaining) + 1):
                        new_seq = remaining[:j] + segment + remaining[j:]
                        if new_seq == best_seq:
                            continue
                        new_expanded, new_dist = try_expand(new_seq)
                        if new_expanded is not None and new_dist < best_dist - 1e-9:
                            best_seq = new_seq
                            best_expanded = new_expanded
                            best_dist = new_dist
                            improved = True
                            break
    
        return best_seq
    
    # Optimize each route
    if time.time() < deadline - 5:
        optimized_routes = []
        for route in routes:
            cust_seq = extract_customers(route)
            opt_seq = two_opt_and_oropt(cust_seq)
            opt_expanded, _ = try_expand(opt_seq)
            if opt_expanded is not None:
                optimized_routes.append(opt_expanded)
            else:
                optimized_routes.append(route)
        routes = optimized_routes
    
    # Phase 3: Inter-route swaps
    if time.time() < deadline - 3 and len(routes) > 1:
        improved = True
        max_swap_iters = 5
        swap_iter = 0
        while improved and swap_iter < max_swap_iters and time.time() < deadline - 3:
            improved = False
            swap_iter += 1
            
            for i in range(len(routes)):
                if time.time() >= deadline - 3:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() >= deadline - 3:
                        break
                    
                    cust_i = extract_customers(routes[i])
                    cust_j = extract_customers(routes[j])
                    
                    if len(cust_i) == 0 or len(cust_j) == 0:
                        continue
                    
                    # Try swapping one customer from each route
                    for c_i in cust_i:
                        for c_j in cust_j:
                            if demands[c_i] + sum(demands[c] for c in cust_j if c != c_j) > capacity:
                                continue
                            if demands[c_j] + sum(demands[c] for c in cust_i if c != c_i) > capacity:
                                continue
                            
                            new_cust_i = [c for c in cust_i if c != c_i] + [c_j]
                            new_cust_j = [c for c in cust_j if c != c_j] + [c_i]
                            
                            exp_i, dist_i = try_expand(new_cust_i)
                            if exp_i is None:
                                continue
                            exp_j, dist_j = try_expand(new_cust_j)
                            if exp_j is None:
                                continue
                            
                            # Get current distances
                            _, cur_dist_i = try_expand(cust_i)
                            _, cur_dist_j = try_expand(cust_j)
                            
                            if dist_i + dist_j < cur_dist_i + cur_dist_j - 1e-9:
                                routes[i] = exp_i
                                routes[j] = exp_j
                                improved = True
    
    return routes

def plan(instance):
    start_time = time.time()
    # Leave 10s safety margin for the evaluator's 150s limit
    deadline = start_time + 140
    
    routes = _solve_instance(instance, deadline)
    
    return routes
# EVOLVE-BLOCK-END
