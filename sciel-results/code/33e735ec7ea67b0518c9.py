import math
import time
import random
import numpy as np

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0

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

    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))
    station_set = set(stations)

    # Precompute distance matrix using numpy
    cx = coords[:, 0]
    cy = coords[:, 1]
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx * dx + dy * dy)

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_set

    # Fast route evaluation: returns (total_dist, valid)
    def check_route(route):
        if not route:
            return 0.0, True

        total_dist = 0.0
        t = 0.0
        b = battery_max
        load = 0.0
        prev = 0

        for node in route:
            if is_customer(node):
                load += demands[node]
        if load > capacity + _EPS:
            return float('inf'), False

        for node in route:
            d = dist[prev, node]
            total_dist += d
            t += d / speed
            b -= d

            if b < -_EPS:
                return float('inf'), False

            if is_station(node):
                t += (battery_max - b) * recharge_rate
                b = battery_max
            elif is_customer(node):
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + _EPS:
                    return float('inf'), False
                t += service[node]

            prev = node

        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        t += d / speed
        b -= d

        if b < -_EPS:
            return float('inf'), False
        if t > horizon + _EPS:
            return float('inf'), False

        return total_dist, True

    # ============ CONSTRUCTION: Nearest Neighbor with Time Windows ============
    def construct_nn():
        routes = []
        unvisited = set(cust_ids)
        
        # Sort by due date for initial route seeding
        sorted_custs = sorted(cust_ids, key=lambda c: (due[c], ready[c]))
        
        while unvisited:
            # Pick the customer with earliest due date as seed
            seed = sorted_custs[0] if sorted_custs else None
            if seed is None or seed not in unvisited:
                seed = next(iter(unvisited))
            
            route = [seed]
            unvisited.remove(seed)
            load = demands[seed]
            current_node = 0
            t = 0.0
            b = battery_max
            
            # Build route greedily
            while unvisited:
                best_cust = None
                best_cost = float('inf')
                
                for c in unvisited:
                    if load + demands[c] > capacity + _EPS:
                        continue
                    
                    # Try inserting c at the end of current route
                    d_go = dist[current_node, c]
                    d_back = dist[c, 0]
                    
                    t_arr = t + d_go / speed
                    b_arr = b - d_go
                    
                    if b_arr < -_EPS:
                        continue
                    
                    # Check if we need to charge before going to c
                    # We can optionally visit a station, but for simplicity in NN, 
                    # we just check direct feasibility and let local search fix it
                    # Actually, let's just check if direct is feasible in time and battery
                    # For battery, we must be able to go back to depot after service
                    t_after_service = max(t_arr, ready[c]) + service[c]
                    
                    if max(t_arr, ready[c]) > due[c] + _EPS:
                        continue
                    
                    # Check battery for return
                    b_after_service = b_arr  # service doesn't consume battery in this model usually, but let's check return
                    # The model says "d uses d battery". Service time is added to time.
                    # So battery is only consumed by distance.
                    b_return = b_after_service - d_back
                    if b_return < -_EPS:
                        continue
                    
                    t_return = t_after_service + d_back / speed
                    if t_return > horizon + _EPS:
                        continue
                        
                    # Cost heuristic: distance + time penalty
                    cost = d_go + d_back
                    # Prefer customers that are close and fit well time-wise
                    if cost < best_cost:
                        best_cost = cost
                        best_cust = c
                        
                if best_cust is not None:
                    route.append(best_cust)
                    unvisited.remove(best_cust)
                    load += demands[best_cust]
                    
                    # Update state
                    d_go = dist[current_node, best_cust]
                    t += d_go / speed
                    b -= d_go
                    
                    t = max(t, ready[best_cust])
                    t += service[best_cust]
                    current_node = best_cust
                else:
                    break
                    
            routes.append(route)
            
            # Remove served from sorted list
            sorted_custs = [c for c in sorted_custs if c in unvisited]
            
        return routes

    # ============ LOCAL SEARCH ============
    def local_search(routes, time_limit=None):
        if time_limit is None:
            time_limit = DEADLINE
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Intra-route 2-Opt
            for ri in range(len(routes)):
                if time.time() > time_limit: break
                route = routes[ri]
                n = len(route)
                if n < 2: continue
                
                cur_cost, cur_valid = check_route(route)
                if not cur_valid:
                    continue
                
                best_route = list(route)
                best_cost = cur_cost
                route_improved = True
                max_iter = 5  # Limit iterations for speed
                while route_improved and max_iter > 0:
                    route_improved = False
                    max_iter -= 1
                    for i in range(n - 1):
                        if time.time() > time_limit: break
                        for j in range(i + 1, n):
                            cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                            c, v = check_route(cand)
                            if v and c < best_cost - 1e-7:
                                best_cost = c
                                best_route = cand
                                route_improved = True
                                
            if best_cost < cur_cost - 1e-7:
                routes[ri] = best_route
                improved = True
            
            # 2. Inter-route Relocate (Or-opt)
            if time.time() > time_limit: break
            for ri in range(len(routes)):
                if time.time() > time_limit: break
                for ci in range(len(routes[ri])):
                    if time.time() > time_limit: break
                    cust = routes[ri][ci]
                    temp_from = routes[ri][:]
                    del temp_from[ci]
                    c_from, v_from = check_route(temp_from)
                    if not v_from:
                        continue
                    
                    for rj in range(len(routes)):
                        if rj == ri: continue
                        if time.time() > time_limit: break
                        for cj in range(len(routes[rj]) + 1):
                            if time.time() > time_limit: break
                            cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                            c_to, v_to = check_route(cand_to)
                            if not v_to:
                                continue
                            
                            load_to = sum(demands[n] for n in cand_to if is_customer(n))
                            if load_to > capacity + _EPS:
                                continue
                            
                            old_cost = c_from + check_route(routes[rj])[0]
                            new_cost = c_from + c_to
                            if new_cost < old_cost - 1e-7:
                                routes[ri] = temp_from
                                routes[rj] = cand_to
                                improved = True
                                ri = -1
                                break
                        if ri == -1: break
                    if ri == -1: break
                if ri == -1: break
            
            routes = [r for r in routes if r]
            
            # 3. Inter-route Swap
            if time.time() > time_limit: break
            swapped = False
            for ri in range(len(routes)):
                if swapped or time.time() > time_limit: break
                for rj in range(ri + 1, len(routes)):
                    if swapped or time.time() > time_limit: break
                    for ci in range(len(routes[ri])):
                        if swapped or time.time() > time_limit: break
                        cust1 = routes[ri][ci]
                        for cj in range(len(routes[rj])):
                            if swapped or time.time() > time_limit: break
                            cust2 = routes[rj][cj]
                            
                            load_i = sum(demands[n] for n in routes[ri] if is_customer(n))
                            load_j = sum(demands[n] for n in routes[rj] if is_customer(n))
                            new_load_i = load_i - demands[cust1] + demands[cust2]
                            new_load_j = load_j - demands[cust2] + demands[cust1]
                            
                            if new_load_i > capacity + _EPS or new_load_j > capacity + _EPS:
                                continue
                                
                            new_route_i = routes[ri][:]
                            new_route_i[ci] = cust2
                            new_route_j = routes[rj][:]
                            new_route_j[cj] = cust1
                            
                            c_i, v_i = check_route(new_route_i)
                            c_j, v_j = check_route(new_route_j)
                            
                            if v_i and v_j:
                                old_c_i = check_route(routes[ri])[0]
                                old_c_j = check_route(routes[rj])[0]
                                if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                    routes[ri] = new_route_i
                                    routes[rj] = new_route_j
                                    improved = True
                                    swapped = True
                                    break
                    if swapped: break
            if swapped: break
            
        return routes

    # ============ ITERATED LOCAL SEARCH ============
    def iterated_local_search():
        current_routes = construct_nn()
        current_routes = local_search(current_routes)
        
        best_routes = [r[:] for r in current_routes]
        best_cost = sum(check_route(r)[0] for r in best_routes)
        
        while time.time() < DEADLINE:
            # Perturb: Remove K customers
            K = random.randint(2, 5)
            all_customers = [c for r in current_routes for c in r if is_customer(c)]
            if len(all_customers) < K:
                break
            
            removed = random.sample(all_customers, K)
            
            for c in removed:
                for i, r in enumerate(current_routes):
                    if c in r:
                        current_routes[i].remove(c)
                        break
            
            current_routes = [r for r in current_routes if r]
            
            # Repair: Insert removed customers
            for c in removed:
                best_route_idx = -1
                best_pos = -1
                best_cost_increase = float('inf')
                
                for ri, r in enumerate(current_routes):
                    for pos in range(len(r) + 1):
                        cand = r[:pos] + [c] + r[pos:]
                        cost, valid = check_route(cand)
                        if valid:
                            old_cost = check_route(r)[0]
                            increase = cost - old_cost
                            if increase < best_cost_increase:
                                best_cost_increase = increase
                                best_route_idx = ri
                                best_pos = pos
                
                if best_route_idx != -1:
                    current_routes[best_route_idx] = current_routes[best_route_idx][:best_pos] + [c] + current_routes[best_route_idx][best_pos:]
                else:
                    current_routes.append([c])
            
            current_routes = local_search(current_routes)
            
            curr_cost = sum(check_route(r)[0] for r in current_routes)
            if curr_cost < best_cost - 1e-7:
                best_cost = curr_cost
                best_routes = [r[:] for r in current_routes]

        return best_routes

    # Run
    routes = iterated_local_search()
    
    # Final validation and fallback
    total_dist = 0.0
    served = set()
    valid = True
    for r in routes:
        c, v = check_route(r)
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
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]

    return routes
