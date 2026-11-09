# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt
from itertools import combinations

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

def _total_distance(route_lists, dist):
    """Sum of expanded distances for all routes."""
    total = 0.0
    for rl in route_lists:
        total += rl[1]
    return total

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
    routes = [] # list of [customer_list, total_demand, expanded_route, total_dist]
    
    for c in customers:
        placed = False
        best_idx = -1
        best_pos = -1
        best_new_dist = float('inf')
        best_new_cust = None
        best_new_exp = None
        
        for i, r in enumerate(routes):
            if r[1] + demands[c] > capacity:
                continue
            cust_list = r[0]
            for pos in range(len(cust_list) + 1):
                new_cust = cust_list[:pos] + [c] + cust_list[pos:]
                exp, d_exp = _expand_route(new_cust, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                if exp is not None:
                    if d_exp < best_new_dist:
                        best_new_dist = d_exp
                        best_idx = i
                        best_pos = pos
                        best_new_cust = new_cust
                        best_new_exp = exp
        
        if best_idx != -1:
            routes[best_idx][0] = best_new_cust
            routes[best_idx][1] += demands[c]
            routes[best_idx][2] = best_new_exp
            routes[best_idx][3] = best_new_dist
            placed = True
        else:
            exp, d_exp = _expand_route([c], dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
            if exp is not None:
                routes.append([[c], demands[c], exp, d_exp])
                placed = True
            else:
                routes.append([[c], demands[c], None, float('inf')])
    
    # Phase 2: Local Search (Best Improvement)
    def get_total_dist():
        return sum(r[3] for r in routes if r[3] < float('inf'))
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        best_gain = 1e-9
        best_move = None
        
        current_total = get_total_dist()
        
        # 1. Try Relocate
        for i in range(len(routes)):
            if time.time() - start_time > time_limit:
                break
            seq_i = routes[i][0]
            for k in range(len(seq_i)):
                c = seq_i[k]
                new_seq_i = seq_i[:k] + seq_i[k+1:]
                
                if new_seq_i:
                    exp_i, d_i = _expand_route(new_seq_i, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                    if exp_i is None:
                        continue
                else:
                    exp_i, d_i = [], 0.0
                
                for j in range(len(routes)):
                    if i == j:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    seq_j = routes[j][0]
                    if routes[j][1] + demands[c] > capacity:
                        continue
                    
                    for p in range(len(seq_j) + 1):
                        new_seq_j = seq_j[:p] + [c] + seq_j[p:]
                        exp_j, d_j = _expand_route(new_seq_j, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                        if exp_j is None:
                            continue
                        
                        # Calculate new total distance
                        # Sum of all routes except i and j, plus new d_i and d_j
                        others_sum = 0.0
                        for idx in range(len(routes)):
                            if idx != i and idx != j:
                                others_sum += routes[idx][3]
                        
                        new_total = d_i + d_j + others_sum
                        gain = current_total - new_total
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('relocate', i, k, j, p, new_seq_i, new_seq_j, exp_i, d_i, exp_j, d_j, c)
        
        # 2. Try 2-Opt within routes
        for i in range(len(routes)):
            if time.time() - start_time > time_limit:
                break
            seq = routes[i][0]
            if len(seq) < 2:
                continue
            for a in range(len(seq) - 1):
                for b in range(a + 1, len(seq)):
                    new_seq = seq[:a] + seq[a:b+1][::-1] + seq[b+1:]
                    exp, d_exp = _expand_route(new_seq, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                    if exp is not None:
                        others_sum = sum(r[3] for idx, r in enumerate(routes) if idx != i)
                        new_total = d_exp + others_sum
                        gain = current_total - new_total
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('2opt', i, a, b, new_seq, exp, d_exp)
        
        if best_move:
            improved = True
            if best_move[0] == 'relocate':
                _, i, k, j, p, new_seq_i, new_seq_j, exp_i, d_i, exp_j, d_j, c = best_move
                # Update route i
                if new_seq_i:
                    routes[i] = [new_seq_i, routes[i][1] - demands[c], exp_i, d_i]
                else:
                    # Remove route i
                    # Shift indices if j > i
                    del routes[i]
                    if j > i:
                        j -= 1
                    # Update route j (now at index j)
                    routes[j] = [new_seq_j, routes[j][1] + demands[c], exp_j, d_j]
                # If new_seq_i was not empty, we updated i in place.
                # If new_seq_i was empty, we deleted i.
                # We need to handle the case where new_seq_i is not empty carefully.
                if new_seq_i:
                    # i is still valid
                    # j is valid
                    routes[j] = [new_seq_j, routes[j][1] + demands[c], exp_j, d_j]
                    
            elif best_move[0] == '2opt':
                _, i, a, b, new_seq, exp, d_exp = best_move
                routes[i][0] = new_seq
                routes[i][2] = exp
                routes[i][3] = d_exp
    
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
