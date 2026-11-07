"""EV Routing: Nearest-Neighbor Construction with Station Insertion + Multi-Pass Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

DEADLINE = 110.0  # Safety margin before 120s limit

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
    
    # Pre-compute distance matrix
    n_nodes = len(coords)
    dist_mat = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist_mat[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    
    def _validate_route(route):
        """Validate a route with time windows, battery, capacity, horizon."""
        total_demand = sum(demands[i] for i in route)
        if total_demand > cap:
            return False
            
        t = 0.0
        batt = batt_max
        prev = 0
        
        for i in route:
            d = dist_mat[prev, i]
            t += d / speed
            batt -= d
            if batt < -1e-9:
                return False
                
            if i in stations:
                wait = (batt_max - batt) * r_rate
                t += wait
                batt = batt_max
            else:
                if t < ready[i]:
                    t = ready[i]
                if t > due[i] + 1e-9:
                    return False
                t += service[i]
                
            prev = i
            
        d = dist_mat[prev, 0]
        t += d / speed
        batt -= d
        if batt < -1e-9:
            return False
        if t > horizon + 1e-9:
            return False
        return True

    def _route_dist(route):
        """Compute total distance of a route including depot legs."""
        if not route:
            return 0.0
        d = dist_mat[0, route[0]]
        for i in range(len(route) - 1):
            d += dist_mat[route[i], route[i+1]]
        d += dist_mat[route[-1], 0]
        return d

    def _try_insert_station(route, cust_idx, inst):
        """Try to insert a charging station before customer at cust_idx to make battery feasible.
        Returns (new_route, success) or (None, False)."""
        # Find which station is closest to the point where battery runs low
        # Simple approach: try each station between prev and current
        coords_local = coords
        demands_local = demands
        ready_local = ready
        due_local = due
        service_local = service
        cap_local = cap
        batt_max_local = batt_max
        speed_local = speed
        r_rate_local = r_rate
        horizon_local = horizon
        stations_local = stations
        dist_mat_local = dist_mat
        
        # Determine the position in the route where we need to insert a station
        # We need to find where battery drops below 0
        # Simulate the route up to cust_idx and find where battery would go negative
        t = 0.0
        batt = batt_max_local
        prev = 0
        
        # Simulate route up to cust_idx
        for i in range(cust_idx + 1):
            node = route[i]
            d = dist_mat_local[prev, node]
            t += d / speed_local
            batt -= d
            if batt < -1e-9:
                # Need a station before this node
                # Try inserting each station before this node
                for st in stations_local:
                    # Check if inserting station st before node i makes it feasible
                    # New segment: prev -> st -> node
                    d1 = dist_mat_local[prev, st]
                    d2 = dist_mat_local[st, node]
                    batt_at_st = batt + d  # battery before going to node (which was negative)
                    # Actually, battery before going to node is batt, and we need to go prev->st->node
                    # Battery at prev is batt_prev, after going to st: batt_prev - d1
                    # After going to st->node: batt_prev - d1 - d2
                    # But we don't have batt_prev directly, we have batt which is batt_prev - d
                    # So batt_prev = batt + d
                    batt_prev = batt + d
                    batt_at_st = batt_prev - d1
                    if batt_at_st < -1e-9:
                        continue
                    # Time at station
                    t_at_st = t - d / speed_local + d1 / speed_local
                    wait = (batt_max_local - batt_at_st) * r_rate_local
                    t_at_st += wait
                    batt_after_st = batt_max_local
                    # Now go st -> node
                    t_at_node = t_at_st + d2 / speed_local
                    batt_at_node = batt_after_st - d2
                    if batt_at_node < -1e-9:
                        continue
                    
                    # Check if this makes the rest feasible - just validate the whole new route
                    new_route = route[:i] + [st] + route[i:]
                    if _validate_route(new_route):
                        return new_route, True
                return None, False
            if node in stations_local:
                wait = (batt_max_local - batt) * r_rate_local
                t += wait
                batt = batt_max_local
            else:
                if t < ready_local[node]:
                    t = ready_local[node]
                t += service_local[node]
            prev = node
        
        return None, False

    def _nearest_neighbor_construct(unassigned, inst):
        """Construct a route using nearest neighbor heuristic with station insertion."""
        if not unassigned:
            return []
        
        coords_local = coords
        demands_local = demands
        ready_local = ready
        due_local = due
        service_local = service
        cap_local = cap
        batt_max_local = batt_max
        speed_local = speed
        r_rate_local = r_rate
        horizon_local = horizon
        stations_local = stations
        dist_mat_local = dist_mat
        
        # Start with the customer with earliest due time
        start_cust = min(unassigned, key=lambda c: due_local[c])
        route = [start_cust]
        unassigned.remove(start_cust)
        total_demand = demands_local[start_cust]
        
        # Nearest neighbor insertion
        while unassigned:
            if time.time() - start_time > DEADLINE:
                break
            
            # Find the nearest unassigned customer that can be added
            best_cust = None
            best_dist = float('inf')
            best_pos = -1
            
            last_node = route[-1]
            for cust in list(unassigned):
                if total_demand + demands_local[cust] > cap_local:
                    continue
                d = dist_mat_local[last_node, cust]
                if d < best_dist:
                    best_dist = d
                    best_cust = cust
                    best_pos = len(route)
            
            if best_cust is None:
                # Try inserting in the middle of the route
                for cust in list(unassigned):
                    if total_demand + demands_local[cust] > cap_local:
                        continue
                    for k in range(len(route)):
                        d = dist_mat_local[route[k-1] if k > 0 else 0, cust] + dist_mat_local[cust, route[k]] - (dist_mat_local[route[k-1] if k > 0 else 0, route[k]] if k > 0 else dist_mat_local[0, route[k]])
                        if d < best_dist:
                            best_dist = d
                            best_cust = cust
                            best_pos = k
                if best_cust is None:
                    break
            
            # Try to add best_cust at best_pos
            new_route = route[:best_pos] + [best_cust] + route[best_pos:]
            
            # Check if battery is feasible
            if not _validate_route(new_route):
                # Try inserting a station
                new_route2, success = _try_insert_station(new_route, best_pos, inst)
                if success:
                    route = new_route2
                    unassigned.remove(best_cust)
                    total_demand += demands_local[best_cust]
                else:
                    # Try other positions
                    found = False
                    for k in range(len(route) + 1):
                        if k == best_pos:
                            continue
                        test_route = route[:k] + [best_cust] + route[k:]
                        if _validate_route(test_route):
                            route = test_route
                            unassigned.remove(best_cust)
                            total_demand += demands_local[best_cust]
                            found = True
                            break
                        if not _validate_route(test_route):
                            test_route2, success2 = _try_insert_station(test_route, k, inst)
                            if success2:
                                route = test_route2
                                unassigned.remove(best_cust)
                                total_demand += demands_local[best_cust]
                                found = True
                                break
                    if not found:
                        # Can't add this customer, skip
                        continue
            else:
                route = new_route
                unassigned.remove(best_cust)
                total_demand += demands_local[best_cust]
        
        return route

    def _two_opt(route, inst):
        """Improve route using 2-opt."""
        coords_local = coords
        demands_local = demands
        ready_local = ready
        due_local = due
        service_local = service
        cap_local = cap
        batt_max_local = batt_max
        speed_local = speed
        r_rate_local = r_rate
        horizon_local = horizon
        stations_local = stations
        dist_mat_local = dist_mat
        
        improved = True
        max_iter = 100
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    if _validate_route(new_route):
                        old_d = _route_dist(route)
                        new_d = _route_dist(new_route)
                        if new_d < old_d - 1e-6:
                            route = new_route
                            improved = True
        return route

    def _or_opt(route, inst):
        """Improve route using Or-Opt (move 1, 2, or 3 consecutive customers)."""
        coords_local = coords
        demands_local = demands
        ready_local = ready
        due_local = due
        service_local = service
        cap_local = cap
        batt_max_local = batt_max
        speed_local = speed
        r_rate_local = r_rate
        horizon_local = horizon
        stations_local = stations
        dist_mat_local = dist_mat
        
        improved = True
        max_iter = 50
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i+length]
                    rest = route[:i] + route[i+length:]
                    if not rest:
                        continue
                    for k in range(len(rest) + 1):
                        if k >= i and k <= i + length:
                            continue
                        new_route = rest[:k] + segment + rest[k:]
                        if _validate_route(new_route):
                            old_d = _route_dist(route)
                            new_d = _route_dist(new_route)
                            if new_d < old_d - 1e-6:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    def _inter_route_move(routes, inst):
        """Try to move a customer from one route to another to reduce total distance."""
        coords_local = coords
        demands_local = demands
        ready_local = ready
        due_local = due
        service_local = service
        cap_local = cap
        batt_max_local = batt_max
        speed_local = speed
        r_rate_local = r_rate
        horizon_local = horizon
        stations_local = stations
        dist_mat_local = dist_mat
        
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                for k in range(len(routes[i])):
                    cust = routes[i][k]
                    # Try removing from route i
                    route_i_new = routes[i][:k] + routes[i][k+1:]
                    if not _validate_route(route_i_new):
                        continue
                    # Try inserting into route j
                    for pos in range(len(routes[j]) + 1):
                        route_j_new = routes[j][:pos] + [cust] + routes[j][pos:]
                        if _validate_route(route_j_new):
                            old_total = _route_dist(routes[i]) + _route_dist(routes[j])
                            new_total = _route_dist(route_i_new) + _route_dist(route_j_new)
                            if new_total < old_total - 1e-6:
                                routes[i] = route_i_new
                                routes[j] = route_j_new
                                improved = True
                                break
                    if improved:
                        break
            if improved:
                break
        return routes, improved

    # Main algorithm
    customers = list(range(1, n_cust + 1))
    
    # Sort by due time for construction
    customers.sort(key=lambda x: due[x])
    
    # Phase 1: Construct routes using nearest neighbor
    unassigned = set(customers)
    routes = []
    
    while unassigned:
        if time.time() - start_time > DEADLINE:
            break
        
        # Start a new route with the earliest due customer
        start_cust = min(unassigned, key=lambda c: due[c])
        route = [start_cust]
        unassigned.remove(start_cust)
        total_demand = demands[start_cust]
        
        # Nearest neighbor expansion
        changed = True
        while changed and unassigned:
            if time.time() - start_time > DEADLINE:
                break
            changed = False
            last_node = route[-1]
            best_cust = None
            best_score = float('inf')
            
            for cust in list(unassigned):
                if total_demand + demands[cust] > cap:
                    continue
                d = dist_mat[last_node, cust]
                # Also check if we can insert in the middle
                for k in range(len(route)):
                    prev_node = route[k-1] if k > 0 else 0
                    d_mid = dist_mat[prev_node, cust] + dist_mat[cust, route[k]] - dist_mat[prev_node, route[k]]
                    if d_mid < best_score:
                        best_score = d_mid
                        best_cust = cust
                        best_pos = k
                if d < best_score:
                    best_score = d
                    best_cust = cust
                    best_pos = len(route)
            
            if best_cust is None:
                break
            
            # Try to insert best_cust
            new_route = route[:best_pos] + [best_cust] + route[best_pos:]
            if _validate_route(new_route):
                route = new_route
                unassigned.remove(best_cust)
                total_demand += demands[best_cust]
                changed = True
            else:
                # Try with station insertion
                new_route2, success = _try_insert_station(new_route, best_pos, instance)
                if success:
                    route = new_route2
                    unassigned.remove(best_cust)
                    total_demand += demands[best_cust]
                    changed = True
                else:
                    # Try other positions
                    found = False
                    for k in range(len(route) + 1):
                        if k == best_pos:
                            continue
                        test_route = route[:k] + [best_cust] + route[k:]
                        if _validate_route(test_route):
                            route = test_route
                            unassigned.remove(best_cust)
                            total_demand += demands[best_cust]
                            changed = True
                            found = True
                            break
                        if not _validate_route(test_route):
                            test_route2, success2 = _try_insert_station(test_route, k, instance)
                            if success2:
                                route = test_route2
                                unassigned.remove(best_cust)
                                total_demand += demands[best_cust]
                                changed = True
                                found = True
                                break
                    if not found:
                        break
        
        if route:
            routes.append(route)
    
    # Handle any remaining unassigned customers (shouldn't happen, but safety)
    for cust in list(unassigned):
        if time.time() - start_time > DEADLINE:
            break
        # Try to insert into existing routes
        inserted = False
        for idx in range(len(routes)):
            for k in range(len(routes[idx]) + 1):
                new_route = routes[idx][:k] + [cust] + routes[idx][k:]
                if _validate_route(new_route):
                    routes[idx] = new_route
                    unassigned.remove(cust)
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            # Create new route
            if _validate_route([cust]):
                routes.append([cust])
                unassigned.remove(cust)
    
    # Phase 2: Local search
    if time.time() - start_time < DEADLINE - 5:
        for idx in range(len(routes)):
            if time.time() - start_time > DEADLINE:
                break
            routes[idx] = _two_opt(routes[idx], instance)
    
    if time.time() - start_time < DEADLINE - 5:
        for idx in range(len(routes)):
            if time.time() - start_time > DEADLINE:
                break
            routes[idx] = _or_opt(routes[idx], instance)
    
    # Phase 3: Inter-route moves
    if time.time() - start_time < DEADLINE - 3:
        improved = True
        max_inter_iter = 20
        iter_count = 0
        while improved and iter_count < max_inter_iter:
            if time.time() - start_time > DEADLINE:
                break
            iter_count += 1
            routes, improved = _inter_route_move(routes, instance)
            if improved:
                # Re-apply local search after moves
                for idx in range(len(routes)):
                    if time.time() - start_time > DEADLINE:
                        break
                    routes[idx] = _two_opt(routes[idx], instance)
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes

# EVOLVE-BLOCK-END
