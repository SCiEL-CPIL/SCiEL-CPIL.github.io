"""EV routing with time windows, charging stations, and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import time

def plan(instance):
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    depot = 0
    
    # Precompute distance matrix
    dist = cdist(coords, coords)
    
    # Node types: 0 = depot, 1 = customer, 2 = station
    node_type = np.zeros(n_nodes, dtype=np.int8)
    for i in range(1, n_nodes):
        if i <= n_customers:
            node_type[i] = 1  # customer
        elif i in stations:
            node_type[i] = 2  # station
    
    customer_ids = list(range(1, n_customers + 1))
    
    # Precompute useful arrays
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_customer[1:n_customers+1] = True
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    
    # Precompute minimum distance from each node to depot (for battery feasibility)
    min_dist_to_depot = dist[:, depot].copy()
    
    def simulate_route(route):
        """Simulate a route, return (total_distance, feasible, end_time, end_battery)"""
        if not route:
            return 0.0, True, 0.0, battery_max
        total_dist = 0.0
        t = 0.0
        batt = battery_max
        prev = depot
        
        for node in route:
            d = dist[prev, node]
            if d > batt:
                return total_dist + d, False, 0, 0
            batt -= d
            t += d / speed
            total_dist += d
            
            if node_type[node] == 1:  # customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node]:
                    return total_dist, False, 0, 0
                t += service[node]
            elif node_type[node] == 2:  # station
                t += (battery_max - batt) / recharge_rate
                batt = battery_max
            
            prev = node
        
        # Return to depot
        d = dist[prev, depot]
        if d > batt:
            return total_dist + d, False, 0, 0
        batt -= d
        t += d / speed
        total_dist += d
        
        if t > horizon:
            return total_dist, False, 0, 0
        
        return total_dist, True, t, batt
    
    def is_valid_solution(routes):
        """Check if all customers are served exactly once and all routes are feasible."""
        served = set()
        for route in routes:
            for node in route:
                if node_type[node] != 1:
                    return False
                if node in served:
                    return False
                served.add(node)
        if served != set(customer_ids):
            return False
        for route in routes:
            _, feasible, _, _ = simulate_route(route)
            if not feasible:
                return False
        return True
    
    def build_initial_routes():
        """Build initial routes using a sweep/cluster approach with feasibility checks."""
        # Sort customers by angle from depot for initial clustering
        angles = np.arctan2(coords[1:n_customers+1, 1] - coords[depot, 1], 
                           coords[1:n_customers+1, 0] - coords[depot, 0])
        sorted_customers = [1 + i for i in np.argsort(angles)]
        
        # Try to pack customers into routes using a greedy approach
        routes = []
        unvisited = set(customer_ids)
        
        # Determine number of routes based on total demand
        total_demand = sum(demands[c] for c in customer_ids)
        n_routes_est = max(1, int(np.ceil(total_demand / capacity)))
        
        # Try different numbers of routes
        for n_r in range(n_routes_est, n_routes_est + 5):
            # Cluster customers into n_r groups by angle
            groups = [[] for _ in range(n_r)]
            for i, cust in enumerate(sorted_customers):
                group_idx = i % n_r
                groups[group_idx].append(cust)
            
            # Build routes from groups
            candidate_routes = []
            valid = True
            for g in groups:
                if not g:
                    continue
                # Sort within group by distance from depot
                g_sorted = sorted(g, key=lambda x: dist[depot, x])
                
                # Try to build a feasible route
                route = []
                total_d = 0
                curr_node = depot
                curr_time = 0.0
                curr_batt = battery_max
                
                for cust in g_sorted:
                    d = dist[curr_node, cust]
                    if d > curr_batt:
                        # Try to insert a charging station
                        found_station = False
                        for st in sorted(stations, key=lambda s: dist[curr_node, s] + dist[s, cust]):
                            d1 = dist[curr_node, st]
                            d2 = dist[st, cust]
                            if d1 <= curr_batt and d2 <= battery_max:
                                # Check time
                                t1 = curr_time + d1 / speed
                                charge_time = (battery_max - (curr_batt - d1)) / recharge_rate
                                t2 = t1 + charge_time + d2 / speed
                                if t2 < ready[cust]:
                                    t2 = ready[cust]
                                if t2 <= due[cust]:
                                    # Check return
                                    d_ret = dist[cust, depot]
                                    if d_ret <= battery_max:
                                        route.append(st)
                                        route.append(cust)
                                        total_d += demands[cust]
                                        curr_node = cust
                                        curr_time = t2 + service[cust]
                                        curr_batt = battery_max - d2
                                        found_station = True
                                        break
                        
                        if not found_station:
                            valid = False
                            break
                    else:
                        new_batt = curr_batt - d
                        new_time = curr_time + d / speed
                        if new_time < ready[cust]:
                            new_time = ready[cust]
                        if new_time > due[cust]:
                            valid = False
                            break
                        d_ret = dist[cust, depot]
                        if new_batt < d_ret:
                            # Try charging
                            found_station = False
                            for st in sorted(stations, key=lambda s: dist[curr_node, s] + dist[s, cust]):
                                d1 = dist[curr_node, st]
                                d2 = dist[st, cust]
                                if d1 <= curr_batt:
                                    t1 = curr_time + d1 / speed
                                    charge_time = (battery_max - (curr_batt - d1)) / recharge_rate
                                    t2 = t1 + charge_time + d2 / speed
                                    if t2 < ready[cust]:
                                        t2 = ready[cust]
                                    if t2 <= due[cust] and d_ret <= battery_max:
                                        route.append(st)
                                        route.append(cust)
                                        total_d += demands[cust]
                                        curr_node = cust
                                        curr_time = t2 + service[cust]
                                        curr_batt = battery_max - d2
                                        found_station = True
                                        break
                            if not found_station:
                                valid = False
                                break
                        else:
                            route.append(cust)
                            total_d += demands[cust]
                            curr_node = cust
                            curr_time = new_time + service[cust]
                            curr_batt = new_batt
                    
                    if total_d > capacity:
                        valid = False
                        break
                
                if valid:
                    candidate_routes.append(route)
                else:
                    break
            
            if valid and all(len(r) > 0 for r in candidate_routes):
                # Verify all customers are served
                all_served = set()
                for r in candidate_routes:
                    for node in r:
                        if node_type[node] == 1:
                            all_served.add(node)
                if all_served == set(customer_ids):
                    # Verify feasibility
                    all_feasible = True
                    for r in candidate_routes:
                        _, feas, _, _ = simulate_route(r)
                        if not feas:
                            all_feasible = False
                            break
                    if all_feasible:
                        return candidate_routes
        
        # Fallback: nearest neighbor
        unvisited = set(customer_ids)
        routes = []
        while unvisited:
            route = []
            total_demand = 0
            current_node = depot
            current_time = 0.0
            current_battery = battery_max
            
            while unvisited:
                best_customer = None
                best_dist = float('inf')
                best_time = 0.0
                best_battery = 0.0
                
                for cust in unvisited:
                    d = dist[current_node, cust]
                    if d > current_battery:
                        continue
                    if total_demand + demands[cust] > capacity:
                        continue
                    
                    new_battery = current_battery - d
                    new_time = current_time + d / speed
                    
                    if new_time < ready[cust]:
                        new_time = ready[cust]
                    if new_time > due[cust]:
                        continue
                    new_time += service[cust]
                    
                    d_return = dist[cust, depot]
                    if new_battery < d_return:
                        continue
                    
                    if d < best_dist:
                        best_dist = d
                        best_customer = cust
                        best_time = new_time
                        best_battery = new_battery
                
                if best_customer is None:
                    break
                
                route.append(best_customer)
                unvisited.remove(best_customer)
                total_demand += demands[best_customer]
                current_node = best_customer
                current_time = best_time
                current_battery = best_battery
            
            if route:
                routes.append(route)
        
        # Ensure all customers are visited
        if unvisited:
            for cust in unvisited:
                routes.append([cust])
        
        return routes
    
    routes = build_initial_routes()
    
    # Validate initial solution
    if not is_valid_solution(routes):
        routes = [[v] for v in customer_ids]
    
    # Precompute route distances and track them
    def get_route_distance(route):
        if not route:
            return 0.0
        d = dist[depot, route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i], route[i+1]]
        d += dist[route[-1], depot]
        return d
    
    # Local search with incremental distance updates
    start_time = time.time()
    time_limit = 110.0
    
    # Track current distances
    route_distances = [get_route_distance(r) for r in routes]
    
    improved = True
    no_improve_count = 0
    max_no_improve = 3
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        n_routes = len(routes)
        
        # 2-opt within each route
        for r in range(n_routes):
            route = routes[r]
            L = len(route)
            if L < 2:
                continue
            old_dist = route_distances[r]
            
            for i in range(L - 1):
                if (time.time() - start_time) > time_limit:
                    break
                for j in range(i + 1, L):
                    # Calculate new distance for 2-opt reversal
                    # Remove edges: (i-1, i) and (j, j+1), add edges: (i-1, j) and (j+1, i)
                    if i == 0:
                        d_remove1 = dist[depot, route[i]]
                        d_add1 = dist[depot, route[j]]
                    else:
                        d_remove1 = dist[route[i-1], route[i]]
                        d_add1 = dist[route[i-1], route[j]]
                    
                    if j == L - 1:
                        d_remove2 = dist[route[j], depot]
                        d_add2 = dist[route[i], depot]
                    else:
                        d_remove2 = dist[route[j], route[j+1]]
                        d_add2 = dist[route[i], route[j+1]]
                    
                    delta = (d_add1 + d_add2) - (d_remove1 + d_remove2)
                    if delta >= 0:
                        continue
                    
                    # Check capacity (unchanged for 2-opt)
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    # Quick feasibility check
                    d_new, feasible, _, _ = simulate_route(new_route)
                    if feasible and d_new < old_dist - 1e-9:
                        routes[r] = new_route
                        route_distances[r] = d_new
                        old_dist = d_new
                        improved = True
        
        # Swap between routes
        for r1 in range(n_routes):
            if (time.time() - start_time) > time_limit:
                break
            for r2 in range(r1 + 1, n_routes):
                L1 = len(routes[r1])
                L2 = len(routes[r2])
                cap1 = sum(demands[node] for node in routes[r1])
                cap2 = sum(demands[node] for node in routes[r2])
                
                for i1 in range(L1):
                    if (time.time() - start_time) > time_limit:
                        break
                    for i2 in range(L2):
                        n1 = routes[r1][i1]
                        n2 = routes[r2][i2]
                        
                        # Check capacity
                        if cap1 - demands[n1] + demands[n2] > capacity:
                            continue
                        if cap2 - demands[n2] + demands[n1] > capacity:
                            continue
                        
                        # Calculate distance change
                        # Route 1: remove n1, insert n2
                        if L1 == 1:
                            d_remove1 = dist[depot, n1] + dist[n1, depot]
                            d_add1 = dist[depot, n2] + dist[n2, depot]
                        else:
                            prev1 = depot if i1 == 0 else routes[r1][i1-1]
                            next1 = depot if i1 == L1-1 else routes[r1][i1+1]
                            d_remove1 = dist[prev1, n1] + dist[n1, next1]
                            d_add1 = dist[prev1, n2] + dist[n2, next1]
                        
                        if L2 == 1:
                            d_remove2 = dist[depot, n2] + dist[n2, depot]
                            d_add2 = dist[depot, n1] + dist[n1, depot]
                        else:
                            prev2 = depot if i2 == 0 else routes[r2][i2-1]
                            next2 = depot if i2 == L2-1 else routes[r2][i2+1]
                            d_remove2 = dist[prev2, n2] + dist[n2, next2]
                            d_add2 = dist[prev2, n1] + dist[n1, next2]
                        
                        delta = (d_add1 + d_add2) - (d_remove1 + d_remove2)
                        if delta >= 0:
                            continue
                        
                        new_r1 = list(routes[r1])
                        new_r2 = list(routes[r2])
                        new_r1[i1] = n2
                        new_r2[i2] = n1
                        
                        d1, f1, _, _ = simulate_route(new_r1)
                        if not f1:
                            continue
                        d2, f2, _, _ = simulate_route(new_r2)
                        if not f2:
                            continue
                        
                        if d1 + d2 < route_distances[r1] + route_distances[r2] - 1e-9:
                            routes[r1] = new_r1
                            routes[r2] = new_r2
                            route_distances[r1] = d1
                            route_distances[r2] = d2
                            improved = True
        
        # Relocate: move a customer from one route to another
        for r1 in range(n_routes):
            if (time.time() - start_time) > time_limit:
                break
            route1 = routes[r1]
            L1 = len(route1)
            cap1 = sum(demands[node] for node in route1)
            
            for pos1 in range(L1):
                if (time.time() - start_time) > time_limit:
                    break
                cust = route1[pos1]
                
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    route2 = routes[r2]
                    L2 = len(route2)
                    cap2 = sum(demands[node] for node in route2)
                    
                    if cap2 + demands[cust] > capacity:
                        continue
                    
                    # Try inserting cust at each position in route2
                    for pos2 in range(L2 + 1):
                        # Calculate distance change
                        # Remove cust from route1
                        if L1 == 1:
                            d_remove1 = dist[depot, cust] + dist[cust, depot]
                        else:
                            prev1 = depot if pos1 == 0 else route1[pos1-1]
                            next1 = depot if pos1 == L1-1 else route1[pos1+1]
                            d_remove1 = dist[prev1, cust] + dist[cust, next1]
                        
                        # Insert cust into route2
                        if L2 == 0:
                            d_add2 = dist[depot, cust] + dist[cust, depot]
                        else:
                            if pos2 == 0:
                                d_add2 = dist[depot, cust] + dist[cust, route2[0]] - dist[depot, route2[0]]
                            elif pos2 == L2:
                                d_add2 = dist[route2[-1], cust] + dist[cust, depot] - dist[route2[-1], depot]
                            else:
                                d_add2 = dist[route2[pos2-1], cust] + dist[cust, route2[pos2]] - dist[route2[pos2-1], route2[pos2]]
                        
                        delta = (d_add2) - (d_remove1)
                        if delta >= 0:
                            continue
                        
                        new_r1 = route1[:pos1] + route1[pos1+1:]
                        new_r2 = route2[:pos2] + [cust] + route2[pos2:]
                        
                        d1, f1, _, _ = simulate_route(new_r1)
                        if not f1:
                            continue
                        d2, f2, _, _ = simulate_route(new_r2)
                        if not f2:
                            continue
                        
                        if d1 + d2 < route_distances[r1] + route_distances[r2] - 1e-9:
                            routes[r1] = new_r1
                            routes[r2] = new_r2
                            route_distances[r1] = d1
                            route_distances[r2] = d2
                            improved = True
        
        if not improved:
            no_improve_count += 1
        else:
            no_improve_count = 0
        
        if no_improve_count >= max_no_improve:
            break
    
    # Final validation
    if is_valid_solution(routes):
        return routes
    else:
        return [[v] for v in customer_ids]
# EVOLVE-BLOCK-END
