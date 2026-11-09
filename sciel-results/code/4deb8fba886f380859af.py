"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _build_distance_matrix(coords):
    """Precompute Euclidean distance matrix using NumPy."""
    coords = np.array(coords, dtype=np.float64)
    x_diff = coords[:, 0, np.newaxis] - coords[np.newaxis, :, 0]
    y_diff = coords[:, 1, np.newaxis] - coords[np.newaxis, :, 1]
    dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
    return dist_matrix

def _simulate_route(route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route and return (total_distance, is_feasible).
    """
    if not route:
        return 0.0, True
        
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    d_mat = dist_matrix
    dem = demands
    rd = ready
    du = due
    svc = service
    st_set = stations_set
    cap = capacity
    bat_max = battery
    spd = speed
    rc_rate = recharge_rate
    hor = horizon
    
    for node in route:
        d = d_mat[cur_node, node]
        cur_time += d / spd
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return total_dist, False
        
        if node in st_set:
            energy_needed = bat_max - cur_bat
            if energy_needed > 0:
                cur_time += energy_needed * rc_rate
                cur_bat = bat_max
        else:
            if cur_time < rd[node]:
                cur_time = rd[node]
            if cur_time > du[node] + 1e-9:
                return total_dist, False
            cur_time += svc[node]
            load += dem[node]
            if load > cap + 1e-9:
                return total_dist, False
        
        cur_node = node
    
    d = d_mat[cur_node, 0]
    cur_time += d / spd
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return total_dist, False
    if cur_time > hor + 1e-9:
        return total_dist, False
        
    return total_dist, True

def _fast_feasibility(route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """
    Fast feasibility check that bails early on obvious infeasibilities.
    Returns True if likely feasible, False if definitely infeasible.
    """
    if not route:
        return True
    
    # Quick capacity check
    load = 0.0
    for node in route:
        if node not in stations_set:
            load += demands[node]
            if load > capacity + 1e-9:
                return False
    
    # Quick battery check: total distance must be <= battery
    total_d = dist_matrix[0, route[0]]
    for i in range(len(route) - 1):
        total_d += dist_matrix[route[i], route[i+1]]
    total_d += dist_matrix[route[-1], 0]
    if total_d > battery + 1e-9:
        # Could still be feasible if we visit stations, but if no stations in route, infeasible
        has_station = any(n in stations_set for n in route)
        if not has_station:
            return False
    
    # Quick time-window check for each customer
    cur_time = 0.0
    cur_node = 0
    for node in route:
        d = dist_matrix[cur_node, node]
        cur_time += d / speed
        if node not in stations_set:
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False
            cur_time += service[node]
        cur_node = node
    cur_time += dist_matrix[cur_node, 0] / speed
    if cur_time > horizon + 1e-9:
        return False
    
    return True

def _calc_route_dist(route, dist_matrix):
    if not route:
        return 0.0
    d = dist_matrix[0, route[0]]
    for i in range(len(route) - 1):
        d += dist_matrix[route[i], route[i+1]]
    d += dist_matrix[route[-1], 0]
    return d

def _solve_instance(instance, time_limit=45.0):
    start_time = time.time()
    
    coords = instance["coords"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    stations_set = set(stations)
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    dist_matrix = _build_distance_matrix(coords)
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    
    routes = []
    route_loads = []  # track load per route
    
    # Construction: Cheapest Insertion with fast feasibility pre-check
    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_addl_dist = float('inf')
        
        for ri, route in enumerate(routes):
            if route_loads[ri] + demands[cust] > capacity + 1e-9:
                continue
            
            # Fast pre-check: would inserting this customer make total distance too large?
            # Estimate: distance from depot to first, between nodes, last to depot
            for pos in range(len(route) + 1):
                # Quick geometric check: new distance delta
                if pos == 0:
                    old_leg = dist_matrix[0, route[0]] if route else 0
                    new_leg = dist_matrix[0, cust] + (dist_matrix[cust, route[0]] if route else 0)
                    delta = new_leg - old_leg
                elif pos == len(route):
                    old_leg = dist_matrix[route[-1], 0]
                    new_leg = dist_matrix[route[-1], cust] + dist_matrix[cust, 0]
                    delta = new_leg - old_leg
                else:
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                    old_leg = dist_matrix[prev_node, next_node]
                    new_leg = dist_matrix[prev_node, cust] + dist_matrix[cust, next_node]
                    delta = new_leg - old_leg
                
                if delta >= best_addl_dist:
                    continue
                
                new_route = route[:pos] + [cust] + route[pos:]
                if not _fast_feasibility(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                    continue
                
                _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                if not feasible:
                    continue
                
                if delta < best_addl_dist:
                    best_addl_dist = delta
                    best_route_idx = ri
                    best_pos = pos
        
        if best_route_idx >= 0:
            routes[best_route_idx].insert(best_pos, cust)
            route_loads[best_route_idx] += demands[cust]
        else:
            # Start new route
            if _fast_feasibility([cust], dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                _, feasible = _simulate_route([cust], dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                if feasible:
                    routes.append([cust])
                    route_loads.append(demands[cust])
                else:
                    routes.append([cust])
                    route_loads.append(demands[cust])
            else:
                routes.append([cust])
                route_loads.append(demands[cust])

    current_total_dist = sum(_calc_route_dist(r, dist_matrix) for r in routes)
    best_routes = [r[:] for r in routes]
    best_dist = current_total_dist
    
    # Local Search with fast feasibility pre-filter
    improved = True
    while improved and (time.time() - start_time < time_limit * 0.9):
        improved = False
        
        # 1. Intra-route 2-Opt (best improvement per route)
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            if n < 2:
                continue
            
            old_d = _calc_route_dist(route, dist_matrix)
            best_new_route = None
            best_new_d = old_d
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Quick distance delta check
                    # Reversing segment [i..j]
                    if i == 0:
                        a = 0
                    else:
                        a = route[i - 1]
                    if j == n - 1:
                        b = 0
                    else:
                        b = route[j + 1]
                    
                    # Old: dist(a, route[i]) + dist(route[j], b)
                    # New: dist(a, route[j]) + dist(route[i], b)
                    old_conn = dist_matrix[a, route[i]] + dist_matrix[route[j], b]
                    new_conn = dist_matrix[a, route[j]] + dist_matrix[route[i], b]
                    
                    if new_conn >= old_conn - 1e-9:
                        continue
                    
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if not _fast_feasibility(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                        continue
                    
                    _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if not feasible:
                        continue
                    
                    new_d = _calc_route_dist(new_route, dist_matrix)
                    if new_d < best_new_d - 1e-6:
                        best_new_d = new_d
                        best_new_route = new_route
            
            if best_new_route is not None:
                routes[ri] = best_new_route
                current_total_dist += best_new_d - old_d
                improved = True
        
        if improved:
            if current_total_dist < best_dist - 1e-6:
                best_dist = current_total_dist
                best_routes = [r[:] for r in routes]
            continue
        
        # 2. Intra-route Or-opt (segment length 1 and 2)
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            if n < 3:
                continue
            
            old_d = _calc_route_dist(route, dist_matrix)
            best_new_route = None
            best_new_d = old_d
            
            for seg_len in [1, 2]:
                for i in range(n - seg_len + 1):
                    seg = route[i:i+seg_len]
                    rest = route[:i] + route[i+seg_len:]
                    if not rest:
                        continue
                    
                    for pos in range(len(rest) + 1):
                        if pos == i or pos == i + seg_len:
                            continue
                        
                        new_route = rest[:pos] + seg + rest[pos:]
                        if not _fast_feasibility(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                            continue
                        
                        new_d = _calc_route_dist(new_route, dist_matrix)
                        if new_d < best_new_d - 1e-6:
                            _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                            if feasible:
                                best_new_d = new_d
                                best_new_route = new_route
            
            if best_new_route is not None:
                routes[ri] = best_new_route
                current_total_dist += best_new_d - old_d
                improved = True
        
        if improved:
            if current_total_dist < best_dist - 1e-6:
                best_dist = current_total_dist
                best_routes = [r[:] for r in routes]
            continue

        # 3. Inter-route Relocation
        found_improvement = False
        for ri in range(len(routes)):
            for rj in range(len(routes)):
                if ri == rj:
                    continue
                
                route_i = routes[ri]
                route_j = routes[rj]
                
                if not route_i:
                    continue
                
                for pos_i in range(len(route_i)):
                    cust = route_i[pos_i]
                    if cust in stations_set:
                        continue
                    
                    new_route_i = route_i[:pos_i] + route_i[pos_i+1:]
                    if not _fast_feasibility(new_route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                        continue
                    _, feasible_i = _simulate_route(new_route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if not feasible_i:
                        continue
                    
                    best_pos_j = -1
                    best_addl = float('inf')
                    base_d_j = _calc_route_dist(route_j, dist_matrix)
                    
                    for pos_j in range(len(route_j) + 1):
                        # Quick geometric delta
                        if pos_j == 0:
                            if not route_j:
                                delta = dist_matrix[0, cust] + dist_matrix[cust, 0]
                            else:
                                delta = dist_matrix[0, cust] + dist_matrix[cust, route_j[0]] - dist_matrix[0, route_j[0]]
                        elif pos_j == len(route_j):
                            delta = dist_matrix[route_j[-1], cust] + dist_matrix[cust, 0] - dist_matrix[route_j[-1], 0]
                        else:
                            delta = dist_matrix[route_j[pos_j-1], cust] + dist_matrix[cust, route_j[pos_j]] - dist_matrix[route_j[pos_j-1], route_j[pos_j]]
                        
                        if delta >= best_addl:
                            continue
                        
                        new_route_j = route_j[:pos_j] + [cust] + route_j[pos_j:]
                        
                        load_j = sum(demands[c] for c in new_route_j if c not in stations_set)
                        if load_j > capacity + 1e-9:
                            continue
                        
                        if not _fast_feasibility(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                            continue
                        
                        _, feasible_j = _simulate_route(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if not feasible_j:
                            continue
                        
                        new_d_j = _calc_route_dist(new_route_j, dist_matrix)
                        addl = new_d_j - base_d_j
                        
                        if addl < best_addl:
                            best_addl = addl
                            best_pos_j = pos_j
                    
                    if best_pos_j >= 0:
                        new_route_j = route_j[:best_pos_j] + [cust] + route_j[best_pos_j:]
                        new_d_i = _calc_route_dist(new_route_i, dist_matrix)
                        old_total = _calc_route_dist(route_i, dist_matrix) + base_d_j
                        new_total = new_d_i + _calc_route_dist(new_route_j, dist_matrix)
                        
                        if new_total < old_total - 1e-6:
                            routes[ri] = new_route_i
                            routes[rj] = new_route_j
                            current_total_dist += new_total - old_total
                            found_improvement = True
                            improved = True
                            break
                if found_improvement:
                    break
            if found_improvement:
                if current_total_dist < best_dist - 1e-6:
                    best_dist = current_total_dist
                    best_routes = [r[:] for r in routes]
                break
        
        if improved:
            continue

        # 4. Inter-route Swap
        found_improvement = False
        for ri in range(len(routes)):
            for rj in range(len(routes)):
                if ri == rj:
                    continue
                
                route_i = routes[ri]
                route_j = routes[rj]
                
                if not route_i or not route_j:
                    continue
                
                for pos_i in range(len(route_i)):
                    cust_i = route_i[pos_i]
                    if cust_i in stations_set:
                        continue
                        
                    for pos_j in range(len(route_j)):
                        cust_j = route_j[pos_j]
                        if cust_j in stations_set:
                            continue
                        
                        # Quick capacity check
                        if abs(demands[cust_i] - demands[cust_j]) > 1e-9:
                            # Loads change, check both
                            pass
                        
                        new_route_i = route_i[:]
                        new_route_j = route_j[:]
                        
                        new_route_i[pos_i] = cust_j
                        new_route_j[pos_j] = cust_i
                        
                        load_i = sum(demands[c] for c in new_route_i if c not in stations_set)
                        load_j = sum(demands[c] for c in new_route_j if c not in stations_set)
                        
                        if load_i > capacity + 1e-9 or load_j > capacity + 1e-9:
                            continue
                        
                        if not _fast_feasibility(new_route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                            continue
                        if not _fast_feasibility(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
                            continue
                        
                        _, feasible_i = _simulate_route(new_route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if not feasible_i:
                            continue
                            
                        _, feasible_j = _simulate_route(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if not feasible_j:
                            continue
                        
                        new_total = _calc_route_dist(new_route_i, dist_matrix) + _calc_route_dist(new_route_j, dist_matrix)
                        old_total = _calc_route_dist(route_i, dist_matrix) + _calc_route_dist(route_j, dist_matrix)
                        
                        if new_total < old_total - 1e-6:
                            routes[ri] = new_route_i
                            routes[rj] = new_route_j
                            current_total_dist += new_total - old_total
                            found_improvement = True
                            improved = True
                            break
                    if found_improvement:
                        break
                if found_improvement:
                    break
            if found_improvement:
                if current_total_dist < best_dist - 1e-6:
                    best_dist = current_total_dist
                    best_routes = [r[:] for r in routes]
                break

    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    try:
        available_time = deadline - time.time()
        search_time = max(1.0, available_time - 5.0)
        
        routes = _solve_instance(instance, time_limit=search_time)
        
        # Final verification
        coords = instance["coords"]
        demands = np.array(instance["demands"], dtype=np.float64)
        ready = np.array(instance["ready"], dtype=np.float64)
        due = np.array(instance["due"], dtype=np.float64)
        service = np.array(instance["service"], dtype=np.float64)
        n_customers = instance["n_customers"]
        stations = instance["stations"]
        stations_set = set(stations)
        capacity = instance["capacity"]
        battery = instance["battery"]
        speed = instance["speed"]
        recharge_rate = instance["recharge_rate"]
        horizon = instance["horizon"]
        
        dist_matrix = _build_distance_matrix(coords)
        
        valid_routes = []
        served = set()
        
        for r in routes:
            _, feasible = _simulate_route(r, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
            if feasible:
                valid_routes.append(r)
                for c in r:
                    if c not in stations_set:
                        served.add(c)
        
        all_customers = set(range(1, n_customers + 1))
        unserved = all_customers - served
        
        for c in unserved:
            inserted = False
            for ri, route in enumerate(valid_routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    _, feasible = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if feasible:
                        valid_routes[ri] = new_route
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                valid_routes.append([c])
        
        return valid_routes

    except Exception:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
# EVOLVE-BLOCK-END
