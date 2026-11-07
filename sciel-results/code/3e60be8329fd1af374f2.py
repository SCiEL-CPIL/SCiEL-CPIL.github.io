import math
import time
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
    station_ids = set(stations)

    # Precompute distance matrix
    cx = coords[:, 0]
    cy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i, j] = d
            dist[j, i] = d

    def is_customer(node):
        return 1 <= node <= n_cust

    def is_station(node):
        return node in station_ids

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

    def try_insert_station(route, max_stations=2):
        """
        If route is infeasible due to battery, try inserting stations.
        Returns (new_route, new_cost) or (None, inf)
        """
        total_dist, valid, t, b = check_route(route)
        if valid:
            return route, total_dist

        best_route = None
        best_cost = float('inf')

        # Level 1: Single station insertion
        for s in stations:
            for i in range(len(route) + 1):
                cand = route[:i] + [s] + route[i:]
                c, v, _, _ = check_route(cand)
                if v and c < best_cost:
                    best_cost = c
                    best_route = cand

        if best_route:
            return best_route, best_cost

        if max_stations >= 2:
            # Level 2: Two station insertions
            for s1 in stations:
                for i1 in range(len(route) + 1):
                    cand1 = route[:i1] + [s1] + route[i1:]
                    for s2 in stations:
                        for i2 in range(len(cand1) + 1):
                            cand2 = cand1[:i2] + [s2] + cand1[i2:]
                            c, v, _, _ = check_route(cand2)
                            if v and c < best_cost:
                                best_cost = c
                                best_route = cand2

        if best_route:
            return best_route, best_cost
        return None, float('inf')

    # --- Construction: Nearest Neighbor with Battery Awareness ---
    # Start with one route per customer, then merge using savings with battery repair
    
    routes = [[c] for c in cust_ids]
    
    # First, ensure each single-customer route is feasible (with station if needed)
    for i in range(len(routes)):
        c, v, _, _ = check_route(routes[i])
        if not v:
            rep, _ = try_insert_station(routes[i], max_stations=2)
            if rep:
                routes[i] = rep
            else:
                # Should not happen with battery 300 and max dist ~141
                pass

    # Savings-based merging
    active = [True] * len(routes)
    changed = True
    while changed and time.time() < DEADLINE - 30:  # Leave 30s for local search
        changed = False
        best_saving = 0
        best_i = -1
        best_j = -1
        best_merged = None
        best_order = 0
        
        for i in range(len(routes)):
            if not active[i]: continue
            ci, vi, _, _ = check_route(routes[i])
            if not vi: continue
            for j in range(i + 1, len(routes)):
                if not active[j]: continue
                cj, vj, _, _ = check_route(routes[j])
                if not vj: continue
                
                # Try merging in both orders
                for order in [0, 1]:
                    if order == 0:
                        merged = routes[i] + routes[j]
                    else:
                        merged = routes[j] + routes[i]
                    
                    c_merged, v_merged, _, _ = check_route(merged)
                    if v_merged:
                        saving = ci + cj - c_merged
                        if saving > best_saving:
                            best_saving = saving
                            best_i = i
                            best_j = j
                            best_merged = merged
                            best_order = order
                    else:
                        # Try to repair with station
                        rep, c_rep = try_insert_station(merged, max_stations=1)
                        if rep:
                            saving = ci + cj - c_rep
                            if saving > best_saving:
                                best_saving = saving
                                best_i = i
                                best_j = j
                                best_merged = rep
                                best_order = order

        if best_i != -1:
            routes[best_i] = best_merged
            active[best_j] = False
            changed = True

    routes = [r for i, r in enumerate(routes) if active[i]]
    
    # --- Local Search ---
    
    def get_route_cost(route):
        return check_route(route)[0]
    
    def two_opt(route):
        """Apply 2-opt until no improvement or deadline."""
        n = len(route)
        if n < 2:
            return route
        best_route = list(route)
        best_cost, _, _, _ = check_route(best_route)
        improved = True
        max_passes = 20
        passes = 0
        while improved and passes < max_passes and time.time() < DEADLINE - 10:
            improved = False
            passes += 1
            for i in range(n - 1):
                if time.time() > DEADLINE - 10:
                    break
                for j in range(i + 1, n):
                    cand = best_route[:i] + list(reversed(best_route[i:j+1])) + best_route[j+1:]
                    c, v, _, _ = check_route(cand)
                    if v and c < best_cost - 1e-7:
                        best_cost = c
                        best_route = cand
                        improved = True
        return best_route

    def relocate(routes):
        """Try to relocate a customer from one route to another."""
        improved = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE - 10:
                break
            if not routes[ri]:
                continue
            for ci in range(len(routes[ri])):
                if time.time() > DEADLINE - 10:
                    break
                cust = routes[ri][ci]
                temp_from = routes[ri][:]
                del temp_from[ci]
                c_from, v_from, _, _ = check_route(temp_from)
                if not v_from:
                    continue
                
                for rj in range(len(routes)):
                    if rj == ri or time.time() > DEADLINE - 10:
                        continue
                    for cj in range(len(routes[rj]) + 1):
                        if time.time() > DEADLINE - 10:
                            break
                        cand_to = routes[rj][:cj] + [cust] + routes[rj][cj:]
                        c_to, v_to, _, _ = check_route(cand_to)
                        if not v_to:
                            continue
                        
                        # Capacity check
                        load_to = sum(demands[n] for n in cand_to if is_customer(n))
                        if load_to > capacity + _EPS:
                            continue
                        
                        old_cost = c_from + get_route_cost(routes[rj])
                        new_cost = c_from + c_to
                        if new_cost < old_cost - 1e-7:
                            routes[ri] = temp_from
                            routes[rj] = cand_to
                            improved = True
                            return improved
        return improved

    def swap_customers(routes):
        """Try to swap customers between two routes."""
        improved = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE - 10:
                break
            if not routes[ri]:
                continue
            for rj in range(ri + 1, len(routes)):
                if time.time() > DEADLINE - 10:
                    break
                if not routes[rj]:
                    continue
                for ci in range(len(routes[ri])):
                    if time.time() > DEADLINE - 10:
                        break
                    cust1 = routes[ri][ci]
                    for cj in range(len(routes[rj])):
                        if time.time() > DEADLINE - 10:
                            break
                        cust2 = routes[rj][cj]
                        
                        # Capacity check
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
                        
                        c_i, v_i, _, _ = check_route(new_route_i)
                        c_j, v_j, _, _ = check_route(new_route_j)
                        
                        if v_i and v_j:
                            old_c_i = get_route_cost(routes[ri])
                            old_c_j = get_route_cost(routes[rj])
                            if old_c_i + old_c_j > c_i + c_j + 1e-7:
                                routes[ri] = new_route_i
                                routes[rj] = new_route_j
                                improved = True
                                return improved
        return improved

    def or_opt(routes):
        """Try to move 2-3 consecutive customers from one route to another."""
        improved = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE - 10:
                break
            n = len(routes[ri])
            if n < 2:
                continue
            for ci in range(n):
                if time.time() > DEADLINE - 10:
                    break
                for seg_len in [2, 3]:
                    if ci + seg_len > n:
                        continue
                    segment = routes[ri][ci:ci+seg_len]
                    temp_from = routes[ri][:]
                    del temp_from[ci:ci+seg_len]
                    c_from, v_from, _, _ = check_route(temp_from)
                    if not v_from:
                        continue
                    
                    for rj in range(len(routes)):
                        if rj == ri or time.time() > DEADLINE - 10:
                            continue
                        for cj in range(len(routes[rj]) + 1):
                            if time.time() > DEADLINE - 10:
                                break
                            cand_to = routes[rj][:cj] + segment + routes[rj][cj:]
                            c_to, v_to, _, _ = check_route(cand_to)
                            if not v_to:
                                continue
                            
                            # Capacity check
                            load_to = sum(demands[n] for n in cand_to if is_customer(n))
                            if load_to > capacity + _EPS:
                                continue
                            
                            old_cost = c_from + get_route_cost(routes[rj])
                            new_cost = c_from + c_to
                            if new_cost < old_cost - 1e-7:
                                routes[ri] = temp_from
                                routes[rj] = cand_to
                                improved = True
                                return improved
        return improved

    # Iterate local search moves
    max_iters = 100
    for it in range(max_iters):
        if time.time() > DEADLINE - 15:
            break
        
        any_improved = False
        
        # 2-opt on each route
        for ri in range(len(routes)):
            if time.time() > DEADLINE - 15:
                break
            old_cost, old_valid, _, _ = check_route(routes[ri])
            if not old_valid:
                rep, rcost = try_insert_station(routes[ri], max_stations=2)
                if rep:
                    routes[ri] = rep
                continue
            new_route = two_opt(routes[ri])
            new_cost, new_valid, _, _ = check_route(new_route)
            if new_valid and new_cost < old_cost - 1e-7:
                routes[ri] = new_route
                any_improved = True
        
        # Relocate
        if time.time() < DEADLINE - 15 and relocate(routes):
            any_improved = True
        
        # Swap
        if time.time() < DEADLINE - 15 and swap_customers(routes):
            any_improved = True
        
        # Or-opt
        if time.time() < DEADLINE - 15 and or_opt(routes):
            any_improved = True
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        if not any_improved:
            break

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
        # Fallback: one customer per route
        customers = list(range(1, n_cust + 1))
        customers.sort(key=lambda v: due[v])
        routes = [[v] for v in customers]
        for i in range(len(routes)):
            if not check_route(routes[i])[1]:
                rep, _ = try_insert_station(routes[i], max_stations=2)
                if rep:
                    routes[i] = rep

    return routes
