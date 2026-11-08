"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def _build_distance_matrix(coords):
    """Precompute Euclidean distance matrix."""
    n = len(coords)
    coords = np.array(coords)
    x_diff = coords[:, 0, np.newaxis] - coords[np.newaxis, :, 0]
    y_diff = coords[:, 1, np.newaxis] - coords[np.newaxis, :, 1]
    dist_matrix = np.sqrt(x_diff**2 + y_diff**2)
    return dist_matrix

def _simulate_route(route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route and return (is_valid, total_distance, total_time).
    """
    if not route:
        return True, 0.0, 0.0
        
    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0
    load = 0.0
    
    for node in route:
        d = dist_matrix[cur_node, node]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d
        
        if cur_bat < -1e-9:
            return False, total_dist, cur_time
        
        if node in stations_set:
            # Recharge
            energy_needed = battery - cur_bat
            if energy_needed > 0:
                cur_time += energy_needed * recharge_rate
                cur_bat = battery
        else:
            # Customer
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False, total_dist, cur_time
            cur_time += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return False, total_dist, cur_time
        
        cur_node = node
    
    # Return to depot
    d = dist_matrix[cur_node, 0]
    cur_time += d / speed
    cur_bat -= d
    total_dist += d
    
    if cur_bat < -1e-9:
        return False, total_dist, cur_time
    if cur_time > horizon + 1e-9:
        return False, total_dist, cur_time
        
    return True, total_dist, cur_time

def _solve_instance(instance, time_limit=120.0):
    start_time = time.time()
    
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations_list = instance["stations"]
    stations_set = set(stations_list)
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    dist_matrix = _build_distance_matrix(coords)
    
    customers = list(range(1, n_customers + 1))
    
    # --- Phase 1: Clustering ---
    # Group customers by time windows to ensure feasibility
    # Sort by ready time, then group into clusters of size ~5-8
    customers_sorted = sorted(customers, key=lambda c: ready[c])
    
    clusters = []
    current_cluster = []
    current_cluster_max_due = -1
    current_cluster_min_ready = float('inf')
    current_cluster_load = 0
    
    for c in customers_sorted:
        # Try to add to current cluster
        if not current_cluster:
            current_cluster.append(c)
            current_cluster_max_due = due[c]
            current_cluster_min_ready = ready[c]
            current_cluster_load = demands[c]
        else:
            # Check if adding c keeps the cluster "tight" in time
            # Heuristic: The window of the cluster shouldn't be too wide
            # And load shouldn't exceed capacity
            if current_cluster_load + demands[c] <= capacity:
                # Check time compatibility
                # If the new customer's ready time is after the cluster's due time, we have a conflict
                # Or if the cluster's ready time is after the new customer's due time
                # Actually, since we sort by ready, ready[c] >= current_cluster_min_ready
                # We need to ensure we can serve all.
                # A simple heuristic: keep cluster size small (e.g., 5) or if due time is increasing too fast.
                # Let's just limit cluster size to 5 for now to keep 2-opt fast.
                if len(current_cluster) < 5:
                    current_cluster.append(c)
                    current_cluster_max_due = max(current_cluster_max_due, due[c])
                    current_cluster_min_ready = min(current_cluster_min_ready, ready[c])
                    current_cluster_load += demands[c]
                else:
                    clusters.append(current_cluster)
                    current_cluster = [c]
                    current_cluster_max_due = due[c]
                    current_cluster_min_ready = ready[c]
                    current_cluster_load = demands[c]
            else:
                clusters.append(current_cluster)
                current_cluster = [c]
                current_cluster_max_due = due[c]
                current_cluster_min_ready = ready[c]
                current_cluster_load = demands[c]
                
    if current_cluster:
        clusters.append(current_cluster)
        
    # --- Phase 2: Construct Routes per Cluster ---
    routes = []
    unserved = set(customers)
    
    # Sort clusters by their minimum ready time
    clusters.sort(key=lambda cl: min(ready[c] for c in cl))
    
    for cluster in clusters:
        # Greedy construction for this cluster
        current_route = []
        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        cur_load = 0.0
        
        # Sort cluster customers by due date
        cluster_sorted = sorted(cluster, key=lambda c: due[c])
        
        for c in cluster_sorted:
            if c not in unserved:
                continue
                
            # Check capacity
            if cur_load + demands[c] > capacity + 1e-9:
                continue
                
            # Simulate going to c
            d = dist_matrix[cur_node, c]
            test_bat = cur_bat - d
            test_time = cur_time + d / speed
            
            if test_bat < -1e-9:
                # Try to insert a station before c?
                # For simplicity in construction, skip if battery is too low.
                # We'll rely on local search or fallback.
                continue
                
            if test_time < ready[c]:
                test_time = ready[c]
            if test_time > due[c] + 1e-9:
                continue
            test_time += service[c]
            
            # Check if we can still return to depot (roughly)
            # This is a soft check to avoid building routes that can't close.
            # d_return = dist_matrix[c, 0]
            # if test_time + d_return/speed > horizon: continue # Too strict?
            
            # Accept
            cur_time = test_time
            cur_bat = test_bat
            cur_load += demands[c]
            cur_node = c
            current_route.append(c)
            unserved.discard(c)
            
        # Close the route
        if current_route:
            d_ret = dist_matrix[cur_node, 0]
            test_bat_ret = cur_bat - d_ret
            test_time_ret = cur_time + d_ret / speed
            
            if test_bat_ret >= -1e-9 and test_time_ret <= horizon + 1e-9:
                routes.append(current_route)
            else:
                # Try to insert a station to make it feasible
                best_station = None
                best_station_dist = float('inf')
                for st in stations_list:
                    d_st = dist_matrix[cur_node, st]
                    if cur_bat - d_st >= -1e-9:
                        bat_after_st = battery
                        t_arrive_st = cur_time + d_st / speed
                        t_leave_st = t_arrive_st + (battery - (cur_bat - d_st)) * recharge_rate
                        t_return = t_leave_st + dist_matrix[st, 0] / speed
                        if t_return <= horizon + 1e-9:
                            if d_st < best_station_dist:
                                best_station_dist = d_st
                                best_station = st
                
                if best_station is not None:
                    current_route.append(best_station)
                    routes.append(current_route)
                else:
                    # Fallback: split into single customers
                    for c in current_route:
                        if c in unserved or True: # c is already removed from unserved
                            routes.append([c])
                            # Note: c is already served in unserved logic, but we are adding it to routes.
                            # We need to be careful not to double serve.
                            # Since we discard from unserved when adding to current_route, 
                            # and we are now discarding current_route, we should re-add to unserved?
                            # No, we just create single routes. The "unserved" set is only used for construction.
                            # The final check will verify all customers are in routes.
    
    # Check for any unserved customers (should be none if clusters covered all, but just in case)
    for c in list(unserved):
        routes.append([c])
        unserved.discard(c)
        
    # --- Phase 3: Local Search (Or-opt and 2-Opt) ---
    # We will run local search on each route, and also try to move customers between routes.
    
    # First, optimize each route individually
    for i in range(len(routes)):
        if len(routes[i]) < 2:
            continue
            
        best_route = routes[i]
        best_dist, _, _ = _simulate_route(best_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
        
        improved = True
        while improved and (time.time() - start_time < time_limit * 0.7):
            improved = False
            n = len(best_route)
            
            # 2-Opt
            for k in range(1, n):
                for l in range(k + 1, n):
                    # Reverse segment k..l
                    new_route = best_route[:k] + best_route[k:l+1][::-1] + best_route[l+1:]
                    valid, dist, _ = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if valid and dist < best_dist - 1e-6:
                        best_route = new_route
                        best_dist = dist
                        improved = True
                        break
                if improved:
                    break
            
            # Or-opt (move 1, 2, or 3 consecutive nodes)
            if not improved:
                for move_len in [1, 2, 3]:
                    for i_node in range(n - move_len + 1):
                        segment = best_route[i_node:i_node+move_len]
                        rest = best_route[:i_node] + best_route[i_node+move_len:]
                        
                        # Try inserting at every position in rest
                        for j_node in range(len(rest) + 1):
                            new_route = rest[:j_node] + segment + rest[j_node:]
                            valid, dist, _ = _simulate_route(new_route, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                            if valid and dist < best_dist - 1e-6:
                                best_route = new_route
                                best_dist = dist
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                        
        routes[i] = best_route
        
    # Second, try to move customers between routes (Relocate)
    # This is more expensive, so we do it with a time limit.
    for i in range(len(routes)):
        if not routes[i]:
            continue
        for j in range(len(routes)):
            if i == j:
                continue
            if not routes[j]:
                continue
            
            # Try moving each customer from i to j
            for k in range(len(routes[i])):
                if routes[i][k] in stations_set:
                    continue # Don't move stations for now, or handle carefully
                
                c = routes[i][k]
                # Remove c from i
                route_i = routes[i][:k] + routes[i][k+1:]
                # Try inserting c into j at all positions
                best_pos = -1
                best_dist = float('inf')
                
                # Calculate current dist of i and j
                dist_i, _, _ = _simulate_route(route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                if not dist_i: # invalid
                    continue
                dist_j, _, _ = _simulate_route(routes[j], dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                base_dist = dist_i + dist_j
                
                for pos in range(len(routes[j]) + 1):
                    new_route_j = routes[j][:pos] + [c] + routes[j][pos:]
                    valid, d_new_j, _ = _simulate_route(new_route_j, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                    if valid:
                        # We need to re-check route_i as well? No, route_i is fixed in this inner loop.
                        # But wait, removing c from i might make i invalid if it was the only thing keeping it valid?
                        # _simulate_route checks validity.
                        # So we need to check if route_i is valid.
                        valid_i, d_i, _ = _simulate_route(route_i, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
                        if valid_i:
                            total_d = d_i + d_new_j
                            if total_d < best_dist - 1e-6:
                                best_dist = total_d
                                best_pos = pos
                                
                if best_pos != -1:
                    # Perform move
                    routes[i] = route_i
                    routes[j] = routes[j][:best_pos] + [c] + routes[j][best_pos:]
                    
    # Final cleanup: remove empty routes
    routes = [r for r in routes if r]
    
    return routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    try:
        routes = _solve_instance(instance, time_limit=140.0)
        
        # Final verification
        # If any route is invalid, fallback to single customers
        # We trust _solve_instance to return valid routes, but let's double check.
        # If invalid, we have a problem. The task says "Any broken rule makes the whole result invalid."
        # So we MUST ensure validity.
        
        # Let's do a quick check. If we find an invalid route, we split it into single customers.
        coords = instance["coords"]
        demands = instance["demands"]
        ready = instance["ready"]
        due = instance["due"]
        service = instance["service"]
        n_customers = instance["n_customers"]
        stations_set = set(instance["stations"])
        capacity = instance["capacity"]
        battery = instance["battery"]
        speed = instance["speed"]
        recharge_rate = instance["recharge_rate"]
        horizon = instance["horizon"]
        
        dist_matrix = _build_distance_matrix(coords)
        
        final_routes = []
        all_customers = set(range(1, n_customers + 1))
        served_customers = set()
        
        for r in routes:
            valid, _, _ = _simulate_route(r, dist_matrix, demands, ready, due, service, stations_set, capacity, battery, speed, recharge_rate, horizon)
            if valid:
                final_routes.append(r)
                for node in r:
                    if node not in stations_set:
                        served_customers.add(node)
            else:
                # Split into single customers
                for node in r:
                    if node not in stations_set:
                        final_routes.append([node])
                        served_customers.add(node)
        
        # Check if any customer is missing
        missing = all_customers - served_customers
        for c in missing:
            final_routes.append([c])
            
        return final_routes
    except Exception:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
# EVOLVE-BLOCK-END
