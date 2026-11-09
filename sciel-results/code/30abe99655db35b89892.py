# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, inf

def _expand_route(customer_route, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    """
    Takes a list of customer IDs and returns (node_ids, total_dist) or (None, 0.0).
    """
    if not customer_route:
        return [], 0.0
        
    route = []
    current_node = 0
    bat = battery_cap
    total_time = 0.0
    total_dist = 0.0
    
    for c in customer_route:
        while True:
            d = dist[current_node, c]
            if bat >= d:
                break
            best_s = None
            best_d_s = float('inf')
            for s in stations:
                d_to_s = dist[current_node, s]
                if bat >= d_to_s:
                    if dist[s, c] <= battery_cap:
                        if d_to_s < best_d_s:
                            best_d_s = d_to_s
                            best_s = s
            if best_s is None:
                return None, 0.0
            d_to_s = dist[current_node, best_s]
            total_time += d_to_s / speed
            bat -= d_to_s
            total_dist += d_to_s
            total_time += (battery_cap - bat) * recharge_rate
            bat = battery_cap
            route.append(best_s)
            current_node = best_s
            
        d = dist[current_node, c]
        total_time += d / speed
        bat -= d
        total_dist += d
        start_service = max(total_time, ready[c])
        if start_service > due[c]:
            return None, 0.0
        total_time = start_service + service[c]
        route.append(c)
        current_node = c
        
    while True:
        d = dist[current_node, 0]
        if bat >= d:
            break
        best_s = None
        best_d_s = float('inf')
        for s in stations:
            d_to_s = dist[current_node, s]
            if bat >= d_to_s:
                if dist[s, 0] <= battery_cap:
                    if d_to_s < best_d_s:
                        best_d_s = d_to_s
                        best_s = s
        if best_s is None:
            return None, 0.0
        d_to_s = dist[current_node, best_s]
        total_time += d_to_s / speed
        bat -= d_to_s
        total_dist += d_to_s
        total_time += (battery_cap - bat) * recharge_rate
        bat = battery_cap
        route.append(best_s)
        current_node = best_s
        
    d = dist[current_node, 0]
    total_time += d / speed
    bat -= d
    total_dist += d
    if total_time > horizon:
        return None, 0.0
    return route, total_dist

def _solve_instance(instance, time_limit=120):
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    # Vectorized distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    
    customers = list(range(1, n_customers + 1))
    start_time = time.time()
    
    # Phase 1: Greedy Construction
    # Sort customers by angle from depot to create coherent routes
    angles = np.arctan2(coords[1:, 1], coords[1:, 0])
    sorted_customers = [c for c in sorted(customers, key=lambda c: angles[c-1])]
    
    routes = [] # list of [customer_list, total_demand, expanded_route, total_dist]
    unvisited = set(sorted_customers)
    
    while unvisited:
        # Start a new route with the first unvisited customer in sorted order
        best_c = min(unvisited, key=lambda c: angles[c-1])
        unvisited.remove(best_c)
        
        current_route = [best_c]
        current_demand = demands[best_c]
        current_node = best_c
        
        while unvisited:
            # Find the next customer that fits in capacity and is "close"
            # We prefer customers that are close to current_node and maintain the general direction
            best_next = None
            best_score = float('inf')
            
            for c in unvisited:
                if current_demand + demands[c] > capacity:
                    continue
                d = dist[current_node, c]
                # Simple heuristic: minimize distance
                if d < best_score:
                    best_score = d
                    best_next = c
            
            if best_next is None:
                break
            
            unvisited.remove(best_next)
            current_route.append(best_next)
            current_demand += demands[best_next]
            current_node = best_next
        
        # Expand the route
        exp, d_exp = _expand_route(current_route, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
        if exp is not None:
            routes.append([current_route, current_demand, exp, d_exp])
        else:
            # If infeasible, try to split or handle individually
            for c in current_route:
                exp_c, d_exp_c = _expand_route([c], dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                if exp_c is not None:
                    routes.append([[c], demands[c], exp_c, d_exp_c])
                else:
                    routes.append([[c], demands[c], None, float('inf')])
    
    # Phase 2: Local Search (First Improvement)
    # This is much faster than Best Improvement because it applies the first move that improves the solution
    # and restarts the search. This allows exploring many more moves in the same time.
    
    def get_total_dist():
        return sum(r[3] for r in routes if r[3] < float('inf'))
    
    improved = True
    no_improve_count = 0
    max_no_improve = 50 # Allow more restarts since each pass is faster
    
    while improved and (time.time() - start_time < time_limit):
        improved = False
        current_total = get_total_dist()
        
        # 1. Try Relocate (First Improvement)
        for i in range(len(routes)):
            if time.time() - start_time > time_limit:
                break
            if improved:
                break
            seq_i = routes[i][0]
            for k in range(len(seq_i)):
                if time.time() - start_time > time_limit or improved:
                    break
                c = seq_i[k]
                new_seq_i = seq_i[:k] + seq_i[k+1:]
                
                if new_seq_i:
                    exp_i, d_i = _expand_route(new_seq_i, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                    if exp_i is None:
                        continue
                else:
                    exp_i, d_i = [], 0.0
                
                for j in range(len(routes)):
                    if i == j or improved:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    seq_j = routes[j][0]
                    if routes[j][1] + demands[c] > capacity:
                        continue
                    
                    for p in range(len(seq_j) + 1):
                        if time.time() - start_time > time_limit or improved:
                            break
                        new_seq_j = seq_j[:p] + [c] + seq_j[p:]
                        exp_j, d_j = _expand_route(new_seq_j, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                        if exp_j is None:
                            continue
                        
                        # Calculate new total distance
                        others_sum = 0.0
                        for idx in range(len(routes)):
                            if idx != i and idx != j:
                                others_sum += routes[idx][3]
                        
                        new_total = d_i + d_j + others_sum
                        if new_total < current_total - 1e-9:
                            # Apply move
                            if new_seq_i:
                                routes[i] = [new_seq_i, routes[i][1] - demands[c], exp_i, d_i]
                            else:
                                del routes[i]
                                if j > i:
                                    j -= 1
                                routes[j] = [new_seq_j, routes[j][1] + demands[c], exp_j, d_j]
                            if new_seq_i:
                                routes[j] = [new_seq_j, routes[j][1] + demands[c], exp_j, d_j]
                            improved = True
                            break
        
        # 2. Try 2-Opt within routes (First Improvement)
        if not improved:
            for i in range(len(routes)):
                if time.time() - start_time > time_limit or improved:
                    break
                seq = routes[i][0]
                if len(seq) < 2:
                    continue
                for a in range(len(seq) - 1):
                    if time.time() - start_time > time_limit or improved:
                        break
                    for b in range(a + 1, len(seq)):
                        if time.time() - start_time > time_limit or improved:
                            break
                        new_seq = seq[:a] + seq[a:b+1][::-1] + seq[b+1:]
                        exp, d_exp = _expand_route(new_seq, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                        if exp is not None:
                            others_sum = sum(r[3] for idx, r in enumerate(routes) if idx != i)
                            new_total = d_exp + others_sum
                            if new_total < current_total - 1e-9:
                                routes[i][0] = new_seq
                                routes[i][2] = exp
                                routes[i][3] = d_exp
                                improved = True
                                break
        
        # 3. Try Swap (Exchange two customers between two routes)
        if not improved:
            for i in range(len(routes)):
                if time.time() - start_time > time_limit or improved:
                    break
                seq_i = routes[i][0]
                for k in range(len(seq_i)):
                    if time.time() - start_time > time_limit or improved:
                        break
                    c1 = seq_i[k]
                    for j in range(len(routes)):
                        if i == j or improved:
                            continue
                        if time.time() - start_time > time_limit:
                            break
                        seq_j = routes[j][0]
                        # Check capacity: removing c1 from i and c2 from j
                        # We need to find a c2 in j such that demands[c2] <= capacity - (routes[i][1] - demands[c1])
                        # and demands[c1] <= capacity - (routes[j][1] - demands[c2])
                        # Actually, simpler: just try all pairs and check capacity of new routes
                        for p in range(len(seq_j)):
                            if time.time() - start_time > time_limit or improved:
                                break
                            c2 = seq_j[p]
                            
                            new_seq_i = seq_i[:k] + [c2] + seq_i[k+1:]
                            new_seq_j = seq_j[:p] + [c1] + seq_j[p+1:]
                            
                            new_demand_i = routes[i][1] - demands[c1] + demands[c2]
                            new_demand_j = routes[j][1] - demands[c2] + demands[c1]
                            
                            if new_demand_i > capacity or new_demand_j > capacity:
                                continue
                            
                            exp_i, d_i = _expand_route(new_seq_i, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                            if exp_i is None:
                                continue
                            exp_j, d_j = _expand_route(new_seq_j, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                            if exp_j is None:
                                continue
                            
                            others_sum = 0.0
                            for idx in range(len(routes)):
                                if idx != i and idx != j:
                                    others_sum += routes[idx][3]
                            
                            new_total = d_i + d_j + others_sum
                            if new_total < current_total - 1e-9:
                                routes[i] = [new_seq_i, new_demand_i, exp_i, d_i]
                                routes[j] = [new_seq_j, new_demand_j, exp_j, d_j]
                                improved = True
                                break
        
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    # Build final result
    final_routes = []
    for r in routes:
        if r[2] is not None:
            final_routes.append(r[2])
        else:
            for c in r[0]:
                exp, _ = _expand_route([c], dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                if exp is not None:
                    final_routes.append(exp)
                else:
                    final_routes.append([c])
    
    return final_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145
    routes = _solve_instance(instance, time_limit=140)
    return routes
# EVOLVE-BLOCK-END
