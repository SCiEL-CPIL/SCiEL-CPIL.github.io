"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances using NumPy
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))

    # Helper to evaluate a route sequence (list of customer/station ids, excluding depot 0)
    # Returns (is_valid, total_distance, total_time)
    def evaluate_route(seq):
        if not seq:
            return True, 0.0, 0.0
        
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        
        for node in seq:
            d = dist[prev][node]
            if bat < d - 1e-9:
                return False, float('inf'), float('inf')
            t += d / speed
            bat -= d
            total_dist += d
            load += demands[node]
            
            if load > capacity:
                return False, float('inf'), float('inf')
            
            if 1 <= node <= n_customers:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False, float('inf'), float('inf')
                t += service[node]
            elif node in stations:
                # Recharge
                deficit = battery_cap - bat
                t += deficit * recharge_rate
                bat = battery_cap
            prev = node
        
        # Return to depot
        d_return = dist[prev][0]
        if bat < d_return - 1e-9:
            return False, float('inf'), float('inf')
        t += d_return / speed
        total_dist += d_return
        
        if t > horizon + 1e-9:
            return False, float('inf'), float('inf')
            
        return True, total_dist, t

    # Function to build a route from a sequence of customers, inserting stations if needed
    # This is used for initial construction and potentially repair
    def build_route_with_charging(cust_seq):
        if not cust_seq:
            return [], 0.0, 0.0
            
        route = []
        t = 0.0
        bat = battery_cap
        load = 0
        prev = 0
        total_dist = 0.0
        
        for cust in cust_seq:
            # Try direct
            d_direct = dist[prev][cust]
            feasible_direct = False
            t_direct = float('inf')
            bat_direct = -1
            
            if bat >= d_direct:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr <= due[cust]:
                    feasible_direct = True
                    t_direct = t_arr + service[cust]
                    bat_direct = bat - d_direct
            
            # Try via stations
            best_t = float('inf')
            best_bat = -1
            best_st = -1
            best_dist_add = float('inf')
            
            if feasible_direct:
                best_t = t_direct
                best_bat = bat_direct
                best_dist_add = d_direct
            
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = deficit * recharge_rate
                t_st += recharge_time
                bat_st = battery_cap
                d_st_to_cust = dist[st][cust]
                if bat_st < d_st_to_cust:
                    continue
                t_cust = t_st + d_st_to_cust / speed
                if t_cust < ready[cust]:
                    t_cust = ready[cust]
                if t_cust > due[cust]:
                    continue
                t_new = t_cust + service[cust]
                bat_new = bat_st - d_st_to_cust
                dist_add = d_to_st + d_st_to_cust
                
                # We prefer the option with lower time, but distance is the objective.
                # However, for construction, we just need a valid route.
                # For local search, we evaluate the full route distance.
                # Here, we pick the first feasible or the one with min time?
                # Let's pick the one that minimizes time to keep options open for future customers.
                if t_new < best_t:
                    best_t = t_new
                    best_bat = bat_new
                    best_st = st
                    best_dist_add = dist_add
            
            if best_t == float('inf'):
                return None, float('inf'), float('inf')
            
            if best_st != -1:
                route.append(best_st)
                route.append(cust)
            else:
                route.append(cust)
                
            t = best_t
            bat = best_bat
            load += demands[cust]
            total_dist += best_dist_add
            prev = cust
        
        # Check return to depot
        d_return = dist[prev][0]
        best_t_return = float('inf')
        best_st_return = -1
        best_dist_return = float('inf')
        
        if bat >= d_return:
            t_return = t + d_return / speed
            if t_return <= horizon:
                best_t_return = t_return
                best_dist_return = d_return
        
        for st in stations:
            d_to_st = dist[prev][st]
            if bat < d_to_st:
                continue
            t_st = t + d_to_st / speed
            bat_at_st = bat - d_to_st
            deficit = battery_cap - bat_at_st
            recharge_time = deficit * recharge_rate
            t_st += recharge_time
            bat_st = battery_cap
            d_st_to_depot = dist[st][0]
            if bat_st < d_st_to_depot:
                continue
            t_depot = t_st + d_st_to_depot / speed
            if t_depot <= horizon:
                if t_depot < best_t_return:
                    best_t_return = t_depot
                    best_st_return = st
                    best_dist_return = d_to_st + d_st_to_depot
        
        if best_t_return == float('inf'):
            return None, float('inf'), float('inf')
            
        if best_st_return != -1:
            route.append(best_st_return)
            
        return route, total_dist + best_dist_return, best_t_return

    # Initial Solution: Savings Heuristic
    customers = list(range(1, n_customers + 1))
    
    # Start with each customer in its own route
    routes = []
    for c in customers:
        route, d, t = build_route_with_charging([c])
        if route:
            routes.append(route)
        else:
            # Fallback if single customer is infeasible (should not happen with given params usually)
            routes.append([c])
            
    # Calculate savings
    # savings[i][j] = d(0,i) + d(0,j) - d(i,j)
    # We only care about customers for savings calculation in the merge logic, 
    # but the actual route might contain stations.
    # For simplicity, we treat the "route" as a list of nodes.
    # Merging two routes A and B means connecting the end of A to the start of B (or reverse).
    # Since routes are sequences, we can try to merge route i and route j by concatenating them.
    
    # A simpler and often effective approach for small N is to start with empty routes and insert customers.
    # Let's restart with a greedy insertion heuristic which is often better for EVRPTW.
    
    routes = []
    remaining = set(customers)
    
    # Sort customers by due date (urgent first)
    sorted_customers = sorted(customers, key=lambda x: due[x])
    
    for c in sorted_customers:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        # Try inserting c into each existing route
        for i, route in enumerate(routes):
            # Try all positions
            for p in range(len(route) + 1):
                new_route_seq = route[:p] + [c] + route[p:]
                # We need to rebuild the route with charging logic to get the correct distance and validity
                # This is expensive, but N is small.
                # Optimization: We can just check if inserting c at p is feasible in the "customer sequence"
                # and then rebuild.
                
                # Let's just rebuild and evaluate
                new_route, new_dist, new_t = build_route_with_charging(new_route_seq)
                if new_route is not None:
                    if new_dist < best_cost:
                        best_cost = new_dist
                        best_route_idx = i
                        best_pos = p
                        best_new_route = new_route
        
        if best_route_idx != -1:
            routes[best_route_idx] = best_new_route
        else:
            # Start a new route
            route, d, t = build_route_with_charging([c])
            if route:
                routes.append(route)
            else:
                routes.append([c])
                
        remaining.discard(c)

    # Local Search: Relocate and 2-Opt
    def get_route_distance(route):
        # Re-evaluate to get accurate distance including depot legs
        # The build_route_with_charging returns distance, but we can also just sum dist
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # Relocate
        for i in range(len(routes)):
            if time.time() > deadline - 5.0: break
            # Get customer sequence of route i
            cust_seq_i = [c for c in routes[i] if 1 <= c <= n_customers]
            for j in range(len(cust_seq_i)):
                if time.time() > deadline - 5.0: break
                c = cust_seq_j = cust_seq_i[j]
                
                # Remove c from route i
                route_i_new_seq = [c2 for c2 in cust_seq_i if c2 != c]
                
                for k in range(len(routes)):
                    if k == i:
                        continue
                    cust_seq_k = [c2 for c2 in routes[k] if 1 <= c2 <= n_customers]
                    
                    # Try inserting c into route k at all positions
                    for pos in range(len(cust_seq_k) + 1):
                        new_seq_k = cust_seq_k[:pos] + [c] + cust_seq_k[pos:]
                        
                        # Rebuild both routes
                        r_i, d_i, t_i = build_route_with_charging(route_i_new_seq)
                        r_k, d_k, t_k = build_route_with_charging(new_seq_k)
                        
                        if r_i is None or r_k is None:
                            continue
                        
                        old_dist = get_route_distance(routes[i]) + get_route_distance(routes[k])
                        new_dist = d_i + d_k
                        
                        if new_dist < old_dist - 1e-6:
                            routes[i] = r_i
                            routes[k] = r_k
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        # 2-Opt
        if time.time() > deadline - 5.0: break
        for i in range(len(routes)):
            if time.time() > deadline - 5.0: break
            cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
            if len(cust_seq) < 2:
                continue
            
            best_seq = cust_seq[:]
            r_best, d_best, t_best = build_route_with_charging(best_seq)
            if r_best is None:
                continue
                
            for a in range(len(best_seq)):
                for b in range(a+1, len(best_seq)):
                    new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                    r_new, d_new, t_new = build_route_with_charging(new_seq)
                    if r_new is not None and d_new < d_best - 1e-6:
                        d_best = d_new
                        best_seq = new_seq[:]
                        r_best = r_new
                        
            if r_best is not None and d_best < get_route_distance(routes[i]) - 1e-6:
                routes[i] = r_best
                improved = True

    # Final validation and cleanup
    valid_routes = []
    covered = set()
    for r in routes:
        # Ensure it's valid
        is_valid, d, t = evaluate_route(r)
        if is_valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
        else:
            # If invalid, try to fix by removing problematic customers? 
            # Or just serve them individually?
            # For now, if a route is invalid, we should split it or serve customers individually.
            # Since we started with valid singletons and only made improving moves, this should be rare.
            # But let's be safe.
            pass

    # Add any missing customers
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            route, d, t = build_route_with_charging([cust])
            if route:
                valid_routes.append(route)
                covered.add(cust)
            else:
                valid_routes.append([cust])
                covered.add(cust)

    return valid_routes
# EVOLVE-BLOCK-END
