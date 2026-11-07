import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin for 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Node IDs: 0 depot, 1..n_cust customers, then stations
    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_set = set(stations)

    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_set

    def check_route(route):
        """
        Returns (total_dist, valid, final_time, final_battery)
        """
        if not route:
            return 0.0, True, 0.0, battery_max
        
        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0
        
        # Check capacity
        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False, 0.0, 0.0

        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d
            
            if b < -_EPS:
                return float('inf'), False, 0.0, 0.0
            
            if is_station(node):
                # Recharge
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                t = max(t, ready[node])
                if t > due[node] + _EPS:
                    return float('inf'), False, 0.0, 0.0
                t += service[node]
            
            prev = node
            
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d
        
        if b < -_EPS:
            return float('inf'), False, 0.0, 0.0
            
        if t > horizon + _EPS:
            return float('inf'), False, 0.0, 0.0
            
        return total_dist, True, t, b

    def route_load(route):
        return sum(demands[n] for n in route if is_customer(n))

    # --- Construction: Greedy Nearest Neighbor with Time Windows ---
    
    # Sort customers by due date for initial ordering heuristic
    cust_sorted = sorted(cust_ids, key=lambda x: due[x])
    
    # Try to build routes by inserting customers greedily
    routes = []
    unvisited = set(cust_sorted)
    
    while unvisited and time.time() < DEADLINE:
        # Start a new route
        route = []
        current_load = 0.0
        t = 0.0
        b = battery_max
        prev = 0
        
        # Try to add customers in order of increasing due date
        candidates = list(unvisited)
        candidates.sort(key=lambda x: due[x])
        
        for cust in candidates:
            if cust not in unvisited:
                continue
            if current_load + demands[cust] > capacity + _EPS:
                continue
            
            # Check if we can reach this customer from current position
            d = dist[prev, cust]
            t_new = t + d / speed
            b_new = b - d
            
            if b_new < -_EPS:
                # Need to go to a station first? For simplicity, skip for now
                # In a more sophisticated version, we'd check station feasibility
                continue
                
            # Check time window
            t_arrive = max(t_new, ready[cust])
            if t_arrive > due[cust] + _EPS:
                continue
                
            t_service = t_arrive + service[cust]
            
            # Check if we can return to depot after this
            d_return = dist[cust, 0]
            t_return = t_service + d_return / speed
            b_return = b_new - d_return
            
            if b_return < -_EPS or t_return > horizon + _EPS:
                continue
                
            # Add customer to route
            route.append(cust)
            unvisited.discard(cust)
            current_load += demands[cust]
            t = t_service
            b = b_new
            prev = cust
            
        if route:
            # Validate the route
            c, v, _, _ = check_route(route)
            if v:
                routes.append(route)
            else:
                # If invalid, add customers individually
                for cust in route:
                    unvisited.add(cust)
                # Re-add as single customer routes
                for cust in route:
                    single = [cust]
                    c2, v2, _, _ = check_route(single)
                    if v2:
                        routes.append(single)
                        unvisited.discard(cust)
                    else:
                        # Try to repair with station
                        repaired = None
                        for s in stations:
                            for pos in range(len(single) + 1):
                                cand = single[:pos] + [s] + single[pos:]
                                c3, v3, _, _ = check_route(cand)
                                if v3:
                                    repaired = cand
                                    break
                            if repaired:
                                break
                        if repaired:
                            routes.append(repaired)
                            unvisited.discard(cust)
                        else:
                            # Last resort: keep as is (will be invalid)
                            routes.append(single)
                            unvisited.discard(cust)
                break  # Stop if we can't place any more
    
    # Ensure all customers are served
    if unvisited:
        for cust in list(unvisited):
            single = [cust]
            c, v, _, _ = check_route(single)
            if not v:
                # Try to repair
                repaired = None
                for s in stations:
                    for pos in range(len(single) + 1):
                        cand = single[:pos] + [s] + single[pos:]
                        c2, v2, _, _ = check_route(cand)
                        if v2:
                            repaired = cand
                            break
                    if repaired:
                        break
                if repaired:
                    routes.append(repaired)
                else:
                    routes.append(single)
            else:
                routes.append(single)
            unvisited.discard(cust)
    
    # --- Local Search ---
    
    route_costs = [check_route(r)[0] for r in routes]

    def try_repair(route):
        c, v, _, _ = check_route(route)
        if v:
            return route, c
        
        best_route = None
        best_cost = float('inf')
        
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c2, v2, _, _ = check_route(cand)
                if v2 and c2 < best_cost:
                    best_cost = c2
                    best_route = cand
                    
        if best_route:
            return best_route, best_cost
        return None, float('inf')

    def optimize_route(route):
        """Apply 2-opt to a single route to minimize distance while maintaining feasibility."""
        n = len(route)
        if n < 2:
            return route, check_route(route)[0]
        
        cur_cost, cur_valid, _, _ = check_route(route)
        if not cur_valid:
            rep, rcost = try_repair(route)
            if rep:
                route = rep
                cur_cost = rcost
                cur_valid = True
            else:
                return route, float('inf')
        
        best_route = list(route)
        best_cost = cur_cost
        
        # First pass: standard 2-opt
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        
        # Second pass: Or-opt (move 1-3 consecutive nodes)
        improved = True
        while improved:
            improved = False
            n_curr = len(best_route)
            for i in range(n_curr):
                for k in range(1, 4):
                    if i + k > n_curr:
                        continue
                    segment = best_route[i:i+k]
                    rest = best_route[:i] + best_route[i+k:]
                    for j in range(len(rest) + 1):
                        cand = rest[:j] + segment + rest[j:]
                        c, v, _, _ = check_route(cand)
                        if v and c < best_cost - 1e-7:
                            best_cost = c
                            best_route = cand
                            improved = True
        
        return best_route, best_cost

    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        
        # 1. Intra-route optimization (2-opt + or-opt)
        for ri in range(len(routes)):
            if time.time() > DEADLINE: break
            route = routes[ri]
            
            new_route, new_cost = optimize_route(route)
            
            if new_cost < route_costs[ri] - 1e-7:
                routes[ri] = new_route
                route_costs[ri] = new_cost
                improved = True
        
        # 2. Inter-route Relocate
        if time.time() > DEADLINE: break
        relocate_done = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE or relocate_done: break
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE or relocate_done: break
                cust = routes[ri][ci]
                
                if not is_customer(cust):
                    continue
                    
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or time.time() > DEADLINE or relocate_done: continue
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE or relocate_done: break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        
                        load_to = route_load(routes[rj]) + demands[cust]
                        if load_to > capacity + _EPS:
                            continue
                            
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        
                        old_cost = route_costs[ri] + route_costs[rj]
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            route_costs[ri] = c_from
                            route_costs[rj] = c_to
                            improved = True
                            relocate_done = True
                            break
                    if relocate_done: break
                if relocate_done: break
            if relocate_done: break
        
        # Remove empty routes
        new_routes = []
        new_costs = []
        for i, r in enumerate(routes):
            if r:
                new_routes.append(r)
                new_costs.append(route_costs[i])
        if len(new_routes) != len(routes):
            routes = new_routes
            route_costs = new_costs
            
        # 3. Inter-route Swap
        if time.time() > DEADLINE: break
        swap_done = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE or swap_done: break
            for rj in range(ri + 1, len(routes)):
                if time.time() > DEADLINE or swap_done: break
                for ci in range(len(routes[ri])):
                    if time.time() > DEADLINE or swap_done: break
                    cust1 = routes[ri][ci]
                    if not is_customer(cust1): continue
                    for cj in range(len(routes[rj])):
                        if time.time() > DEADLINE or swap_done: break
                        cust2 = routes[rj][cj]
                        if not is_customer(cust2): continue
                        
                        load_i = route_load(routes[ri])
                        load_j = route_load(routes[rj])
                        new_load_i = load_i - demands[cust1] + demands[cust2]
                        new_load_j = load_j - demands[cust2] + demands[cust1]
                        
                        if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                            continue
                            
                        new_route_i = routes[ri][:]
                        new_route_i[ci] = cust2
                        new_route_j = routes[rj][:]
                        new_route_j[cj] = cust1
                        
                        c_i, v_i, _, _ = check_route(new_route_i)
                        c_j, v_j, _, _ = check_route(new_route_j)
                        
                        if v_i and v_j:
                            old_c_i = route_costs[ri]
                            old_c_j = route_costs[rj]
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                route_costs[ri] = c_i
                                route_costs[rj] = c_j
                                improved = True
                                swap_done = True
                                break
                if swap_done: break
            if swap_done: break

    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v, _, _ = check_route(r)
        if not v:
            valid = False
            break
        total_dist += c
        for node in r:
            if node in served:
                valid = False
                break
            served.add(node)
        if not valid:
            break
            
    if not valid or served != set(range(1, n_cust + 1)):
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_repair(routes[i])
                if rep:
                    routes[i] = rep

    return routes
