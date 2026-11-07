"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import heapq
import math

def plan(instance):
    start_time = time.time()
    deadline = 145.0 # 5 seconds margin
    
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distances
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    
    customers = list(range(1, n_customers + 1))
    
    # Helper: Get best path from u to v given current battery
    # Returns (time, battery_at_v) or None
    def get_path_info(u, v, curr_bat):
        # Direct
        d = dist[u][v]
        if d <= curr_bat:
            t = d / speed
            b = curr_bat - d
            return (t, b)
        
        # Via stations
        best_t = float('inf')
        best_b = 0
        for s in stations:
            if s == u or s == v:
                continue
            d1 = dist[u][s]
            if d1 > curr_bat:
                continue
            d2 = dist[s][v]
            if d2 > battery_cap:
                continue
            
            # Charge time
            bat_at_s = curr_bat - d1
            charge_time = (battery_cap - bat_at_s) * recharge_rate / speed # speed is 1, but keep for generality
            total_t = d1 / speed + charge_time + d2 / speed
            final_bat = battery_cap - d2
            
            if total_t < best_t:
                best_t = total_t
                best_b = final_bat
        
        if best_t < float('inf'):
            return (best_t, best_b)
        return None

    # Helper: Check if we can return to depot from node u, given time t and battery b
    def is_feasible_return(u, t, b):
        # Direct
        d = dist[u][0]
        if d <= b:
            total_t = t + d / speed
            if total_t <= horizon:
                return True
        
        # Via stations
        for s in stations:
            if s == u:
                continue
            d1 = dist[u][s]
            if d1 > b:
                continue
            d2 = dist[s][0]
            if d2 > battery_cap:
                continue
            
            bat_at_s = b - d1
            charge_time = (battery_cap - bat_at_s) * recharge_rate / speed
            total_t = t + d1 / speed + charge_time + d2 / speed
            if total_t <= horizon:
                return True
        return False

    def is_feasible_route(route):
        """
        route: list of node ids (excluding depot at start/end)
        Returns total distance if feasible, else -1
        """
        curr_node = 0
        curr_time = 0.0
        curr_bat = float(battery_cap)
        curr_load = 0
        
        for c in route:
            if curr_load + demands[c] > capacity:
                return -1
            path_info = get_path_info(curr_node, c, curr_bat)
            if path_info is None:
                return -1
            t, b = path_info
            curr_time += t
            curr_bat = b
            
            # Service
            start_serv = max(curr_time, ready[c])
            if start_serv > due[c]:
                return -1
            curr_time = start_serv + service[c]
            curr_load += demands[c]
            curr_node = c
            
            # Check return feasibility early to prune? 
            # Not strictly necessary for final check, but good for construction.
        
        # Return to depot
        path_info = get_path_info(curr_node, 0, curr_bat)
        if path_info is None:
            return -1
        t, b = path_info
        curr_time += t
        if curr_time > horizon:
            return -1
            
        total_dist = 0
        prev = 0
        for c in route:
            total_dist += dist[prev][c]
            prev = c
        total_dist += dist[prev][0]
        return total_dist

    def two_opt(route):
        """
        Apply 2-opt improvement to a single route.
        Returns improved route.
        """
        if len(route) < 2:
            return route
        
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    # Segment reversal: route[i:j+1]
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    # Check feasibility
                    if is_feasible_route(new_route) != -1:
                        # Compare distance
                        old_dist = is_feasible_route(route)
                        new_dist = is_feasible_route(new_route)
                        if new_dist < old_dist - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # Construction
    unvisited = set(customers)
    routes = []
    
    # Sort customers by due time for initial bias
    unvisited_list = sorted(unvisited, key=lambda x: due[x])
    
    while unvisited:
        if time.time() > deadline:
            break
            
        current_route = []
        curr_node = 0
        curr_time = 0.0
        curr_bat = float(battery_cap)
        curr_load = 0
        
        # Greedy construction
        while True:
            if time.time() > deadline:
                break
                
            best_c = None
            best_score = float('inf')
            best_info = None
            best_dep_time = 0
            best_bat = 0
            
            # Iterate over candidates
            # To speed up, we might want to check nearest first, but with 25 it's fine
            for c in unvisited:
                if curr_load + demands[c] > capacity:
                    continue
                
                path_info = get_path_info(curr_node, c, curr_bat)
                if path_info is None:
                    continue
                
                t, b = path_info
                arr_time = curr_time + t
                
                start_serv = max(arr_time, ready[c])
                if start_serv > due[c]:
                    continue
                
                dep_time = start_serv + service[c]
                
                # Check if we can return to depot from c
                if not is_feasible_return(c, dep_time, b):
                    continue
                
                # Score: prefer closer, and maybe tighter time windows?
                # Let's just use distance for now
                score = dist[curr_node][c]
                
                if score < best_score:
                    best_score = score
                    best_c = c
                    best_info = path_info
                    best_dep_time = dep_time
                    best_bat = b
            
            if best_c is None:
                break
            
            current_route.append(best_c)
            unvisited.remove(best_c)
            curr_node = best_c
            curr_time = best_dep_time
            curr_bat = best_bat
            curr_load += demands[best_c]
            
        if current_route:
            # Verify the constructed route is actually feasible (should be, but double check)
            if is_feasible_route(current_route) != -1:
                routes.append(current_route)
            else:
                # Fallback: serve them one by one if construction failed (shouldn't happen)
                for c in current_route:
                    routes.append([c])
                unvisited.update(current_route) # Re-add? No, we already removed. 
                # Actually, if construction failed, it means the last step broke it.
                # Let's just trust the construction if it passed local checks.
                # The local check `is_feasible_return` ensures the last node can go back.
                # And each step checked capacity and windows.
                # So it should be valid.

    # If time remains, apply 2-opt
    if time.time() < deadline - 5:
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            routes[i] = two_opt(routes[i])
            
    # Fallback: if any customers were not served (due to timeout or failure), serve them one by one
    # The loop above removes from unvisited. If we broke out early, some might remain.
    if unvisited:
        for c in unvisited:
            routes.append([c])
            
    return routes
# EVOLVE-BLOCK-END
