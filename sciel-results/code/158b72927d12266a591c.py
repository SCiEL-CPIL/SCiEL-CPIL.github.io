import numpy as np
import time
import random
import math

def plan(instance):
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Extract data
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    max_battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Distance matrix
    # coords shape (N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    
    # Helper: Simulate a route (list of customer ids)
    # Returns (total_distance, is_feasible)
    def evaluate_route(route):
        if not route:
            return 0.0, True
        
        current_node = 0
        current_time = 0.0
        current_battery = max_battery
        current_load = 0
        total_dist = 0.0
        
        for cust in route:
            # Check capacity
            current_load += demands[cust]
            if current_load > capacity:
                return float('inf'), False
            
            # Try to go from current_node to cust
            # Direct?
            dist_direct = D[current_node, cust]
            
            if dist_direct <= current_battery:
                # Check time
                travel_time = dist_direct / speed
                arrival_time = current_time + travel_time
                if arrival_time > due[cust]:
                    return float('inf'), False
                # Service
                start_service = max(arrival_time, ready[cust])
                end_service = start_service + service[cust]
                
                # Update state
                current_time = end_service
                current_battery -= dist_direct
                total_dist += dist_direct
                current_node = cust
            else:
                # Need to charge
                # Find a station s such that:
                # 1. D[current_node, s] <= current_battery
                # 2. D[s, cust] <= max_battery (since we refill to max)
                # 3. Time constraints met
                best_station = -1
                best_total_dist = float('inf')
                best_end_time = float('inf')
                
                for s in stations:
                    d1 = D[current_node, s]
                    if d1 > current_battery:
                        continue
                    d2 = D[s, cust]
                    if d2 > max_battery:
                        continue
                    
                    # Time to reach station
                    t1 = current_time + d1 / speed
                    # Recharge time
                    energy_needed = max_battery - current_battery
                    recharge_time = energy_needed * recharge_rate
                    t2 = t1 + recharge_time
                    # Time to reach customer
                    t3 = t2 + d2 / speed
                    
                    if t3 > due[cust]:
                        continue
                    
                    # Feasible station
                    # We prefer the one that minimizes total distance? Or just any?
                    # For cost evaluation, we must use the *actual* path taken.
                    # If multiple stations work, which one is "the" path?
                    # The problem implies we choose the path. To minimize distance, we should pick the station that minimizes (d1 + d2).
                    # However, time also matters. If we pick a station that is close but causes a huge wait, is it better?
                    # The cost is distance. So we should pick the station that minimizes d1 + d2, subject to time feasibility.
                    # If time is tight, a station with shorter d1+d2 might arrive earlier, which is good.
                    # So min d1+d2 is a good heuristic for the "optimal" choice given we want to minimize distance.
                    
                    total_d = d1 + d2
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_station = s
                        best_end_time = t3
                        # Store intermediate times if needed, but we just need final state
                        # Actually, we need to update current_time and current_battery correctly.
                        # If we take this station:
                        # current_time becomes max(best_end_time, ready[cust]) + service[cust]
                        # current_battery becomes max_battery - d2
                        # total_dist increases by d1 + d2
                        
                if best_station == -1:
                    return float('inf'), False
                
                # Apply the chosen station path
                # Note: This assumes the "best" station is the one with min distance.
                # Is it possible that a longer path allows earlier arrival which is required?
                # We already checked `t3 > due[cust]`.
                # What if `t3 < ready[cust]`? We wait. That's fine.
                # Distance is fixed for this choice.
                
                # Update state
                total_dist += best_total_dist
                # Time update
                arrival_at_cust = best_end_time
                start_service = max(arrival_at_cust, ready[cust])
                current_time = start_service + service[cust]
                current_battery = max_battery - D[best_station, cust]
                current_node = cust
        
        # Return to depot
        dist_back = D[current_node, 0]
        if dist_back <= current_battery:
            travel_time = dist_back / speed
            arrival_time = current_time + travel_time
            if arrival_time > horizon:
                return float('inf'), False
            total_dist += dist_back
        else:
            # Need to charge on the way back
            # Find station s
            # 1. D[current_node, s] <= current_battery
            # 2. D[s, 0] <= max_battery
            # 3. Time <= horizon
            best_station = -1
            best_total_dist = float('inf')
            best_arrival = float('inf')
            
            for s in stations:
                d1 = D[current_node, s]
                if d1 > current_battery:
                    continue
                d2 = D[s, 0]
                if d2 > max_battery:
                    continue
                
                t1 = current_time + d1 / speed
                energy_needed = max_battery - current_battery
                recharge_time = energy_needed * recharge_rate
                t2 = t1 + recharge_time
                t3 = t2 + d2 / speed
                
                if t3 > horizon:
                    continue
                
                total_d = d1 + d2
                if total_d < best_total_dist:
                    best_total_dist = total_d
                    best_station = s
                    best_arrival = t3
            
            if best_station == -1:
                return float('inf'), False
            
            total_dist += best_total_dist
            # No time check against horizon for the final arrival other than the one above
        
        return total_dist, True

    # Initial Solution: 1 customer per route
    # Then merge greedily
    routes = [[c] for c in customers]
    
    # Calculate initial cost
    def get_total_cost(routes):
        cost = 0
        for r in routes:
            c, feas = evaluate_route(r)
            if not feas:
                return float('inf')
            cost += c
        return cost
    
    current_cost = get_total_cost(routes)
    best_routes = [r[:] for r in routes]
    best_cost = current_cost
    
    # Greedy Merging
    # Sort customers by due time for a better initial start?
    # Let's try to merge routes that are close.
    # We'll do a simple pass: try to merge every pair, keep the best improvement.
    # This is O(N^2) merges, each evaluation is O(N). N=25. 25^2 * 25 = 15k ops. Very fast.
    
    merged = True
    while merged:
        merged = False
        best_delta = 0
        best_i, best_j, best_order = -1, -1, 0
        current_routes_copy = [r[:] for r in routes]
        
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Try merging route j into route i
                # Order 1: i + j
                # Order 2: j + i
                for order in [0, 1]:
                    if order == 0:
                        new_route = routes[i] + routes[j]
                    else:
                        new_route = routes[j] + routes[i]
                    
                    # Check if new_route is feasible
                    new_cost, feas = evaluate_route(new_route)
                    if not feas:
                        continue
                    
                    # Calculate cost change
                    # Old cost = cost(i) + cost(j)
                    # We don't have individual costs stored. Recalculating is expensive if done naively.
                    # But N is small.
                    # Let's just estimate.
                    # Actually, to be precise, we should calculate the cost of the new set of routes.
                    # But that's O(N) per pair.
                    # Let's just compute the cost of the new route and subtract the old ones.
                    # We need a function to get cost of a single route.
                    cost_i, _ = evaluate_route(routes[i])
                    cost_j, _ = evaluate_route(routes[j])
                    old_pair_cost = cost_i + cost_j
                    delta = new_cost - old_pair_cost
                    
                    if delta < best_delta:
                        best_delta = delta
                        best_i, best_j, best_order = i, j, order
                        merged = True
        
        if merged:
            # Perform the best merge
            if best_order == 0:
                new_route = routes[best_i] + routes[best_j]
            else:
                new_route = routes[best_j] + routes[best_i]
            
            # Remove j and i, add new_route
            # Be careful with indices
            routes.remove(routes[best_j])
            routes.remove(routes[best_i])
            routes.append(new_route)
            
            # Update best
            new_total = get_total_cost(routes)
            if new_total < best_cost:
                best_cost = new_total
                best_routes = [r[:] for r in routes]
            
            if time.time() > deadline - 5:
                break

    # Simulated Annealing
    # Start temperature
    T = 10.0
    cooling = 0.99
    min_T = 0.01
    
    # We need to be fast.
    # Moves: Relocate, Swap, 2-Opt
    
    while T > min_T and time.time() < deadline:
        # Pick a move
        move_type = random.choice(['relocate', 'swap', '2opt'])
        
        new_routes = [r[:] for r in routes]
        valid_move = False
        
        if move_type == 'relocate':
            if len(routes) < 2:
                continue
            # Pick source route
            src_idx = random.randint(0, len(routes) - 1)
            if len(routes[src_idx]) <= 1:
                continue
            # Pick customer
            cust_pos = random.randint(0, len(routes[src_idx]) - 1)
            cust = routes[src_idx][cust_pos]
            
            # Pick dest route
            dest_idx = random.randint(0, len(routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(routes) - 1)
            
            # Pick insert position
            ins_pos = random.randint(0, len(routes[dest_idx]))
            
            # Apply
            routes[src_idx].pop(cust_pos)
            routes[dest_idx].insert(ins_pos, cust)
            valid_move = True
            
        elif move_type == 'swap':
            if len(routes) < 2:
                continue
            src_idx = random.randint(0, len(routes) - 1)
            dest_idx = random.randint(0, len(routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(routes) - 1)
            
            if len(routes[src_idx]) == 0 or len(routes[dest_idx]) == 0:
                continue
                
            c1_pos = random.randint(0, len(routes[src_idx]) - 1)
            c2_pos = random.randint(0, len(routes[dest_idx]) - 1)
            
            c1 = routes[src_idx][c1_pos]
            c2 = routes[dest_idx][c2_pos]
            
            routes[src_idx][c1_pos] = c2
            routes[dest_idx][c2_pos] = c1
            valid_move = True
            
        elif move_type == '2opt':
            src_idx = random.randint(0, len(routes) - 1)
            if len(routes[src_idx]) < 2:
                continue
            p1 = random.randint(0, len(routes[src_idx]) - 2)
            p2 = random.randint(p1 + 1, len(routes[src_idx]) - 1)
            
            # Reverse segment
            segment = routes[src_idx][p1:p2+1]
            segment.reverse()
            routes[src_idx][p1:p2+1] = segment
            valid_move = True
            
        if not valid_move:
            continue
            
        # Evaluate
        new_cost = get_total_cost(routes)
        if new_cost == float('inf'):
            # Reject
            routes = new_routes
            continue
            
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = [r[:] for r in routes]
        else:
            routes = new_routes
            
        T *= cooling
        
        # Check time
        if time.time() > deadline - 2:
            break
            
    return best_routes
