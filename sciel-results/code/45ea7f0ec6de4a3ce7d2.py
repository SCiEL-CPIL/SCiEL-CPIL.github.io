# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from math import sqrt

def _expand_route(customer_route, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
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

def _route_distance(seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    if not seq:
        return 0.0
    _, d = _expand_route(seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
    return d

def _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations):
    total = 0.0
    for r in routes:
        total += _route_distance(r, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
    return total

def _local_search(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations, time_limit, start_time):
    """First-improvement local search with relocate, 2-opt, and Or-opt moves."""
    n_customers = len([c for c in range(1, 26)])
    demands_map = {}
    
    improved = True
    max_iters = 500
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        if time.time() - start_time > time_limit:
            break
        
        # Try relocate moves
        for i in range(len(routes)):
            if time.time() - start_time > time_limit:
                break
            seq_i = list(routes[i])
            for k in range(len(seq_i)):
                c = seq_i[k]
                new_seq_i = seq_i[:k] + seq_i[k+1:]
                
                # Check if route i is still valid
                if new_seq_i:
                    exp_i, d_i = _expand_route(new_seq_i, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                    if exp_i is None:
                        continue
                else:
                    d_i = 0.0
                
                for j in range(len(routes)):
                    if i == j:
                        continue
                    if time.time() - start_time > time_limit:
                        break
                    seq_j = list(routes[j])
                    # Simple capacity check: sum of demands
                    demand_i = sum(demands_map.get(x, 0) for x in new_seq_i)
                    demand_j = sum(demands_map.get(x, 0) for x in seq_j)
                    if demand_j + demands_map.get(c, 0) > 100:
                        continue
                    
                    for p in range(len(seq_j) + 1):
                        new_seq_j = seq_j[:p] + [c] + seq_j[p:]
                        exp_j, d_j = _expand_route(new_seq_j, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                        if exp_j is None:
                            continue
                        
                        # Calculate improvement
                        old_total = _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                        # Approximate: just check if this specific move helps
                        if d_i + d_j < 1e-9:
                            # Both empty or one empty
                            pass
                        # Apply move and check
                        old_routes = [list(r) for r in routes]
                        routes[i] = new_seq_i
                        routes[j] = new_seq_j
                        new_total = _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                        if new_total < old_total - 1e-9:
                            improved = True
                            break
                        else:
                            routes[i] = old_routes[i]
                            routes[j] = old_routes[j]
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try 2-opt within routes
        for i in range(len(routes)):
            if time.time() - start_time > time_limit:
                break
            seq = list(routes[i])
            if len(seq) < 2:
                continue
            for a in range(len(seq) - 1):
                for b in range(a + 1, len(seq)):
                    new_seq = seq[:a] + seq[a:b+1][::-1] + seq[b+1:]
                    exp, d_exp = _expand_route(new_seq, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                    if exp is not None:
                        old_total = _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                        routes[i] = new_seq
                        new_total = _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                        if new_total < old_total - 1e-9:
                            improved = True
                            break
                        else:
                            routes[i] = seq
                if improved:
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try Or-opt: move a segment of 2-3 customers
        for seg_len in [2, 3]:
            for i in range(len(routes)):
                if time.time() - start_time > time_limit:
                    break
                seq_i = list(routes[i])
                if len(seq_i) < seg_len:
                    continue
                for k in range(len(seq_i) - seg_len + 1):
                    segment = seq_i[k:k+seg_len]
                    new_seq_i = seq_i[:k] + seq_i[k+seg_len:]
                    
                    if new_seq_i:
                        exp_i, d_i = _expand_route(new_seq_i, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                        if exp_i is None:
                            continue
                    else:
                        d_i = 0.0
                    
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        if time.time() - start_time > time_limit:
                            break
                        seq_j = list(routes[j])
                        demand_i = sum(demands_map.get(x, 0) for x in new_seq_i)
                        demand_j = sum(demands_map.get(x, 0) for x in seq_j)
                        seg_demand = sum(demands_map.get(x, 0) for x in segment)
                        if demand_j + seg_demand > 100:
                            continue
                        
                        for p in range(len(seq_j) + 1):
                            new_seq_j = seq_j[:p] + segment + seq_j[p:]
                            exp_j, d_j = _expand_route(new_seq_j, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                            if exp_j is None:
                                continue
                            
                            old_total = _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                            routes[i] = new_seq_i
                            routes[j] = new_seq_j
                            new_total = _total_distance(routes, dist, ready, due, service, battery_cap, speed, recharge_rate, horizon, stations)
                            if new_total < old_total - 1e-9:
                                improved = True
                                break
                            else:
                                routes[i] = seq_i
                                routes[j] = seq_j
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
            if improved:
                break
    
    return routes

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
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    
    customers = list(range(1, n_customers + 1))
    start_time = time.time()
    
    # Build demand map
    demands_map = {}
    for c in customers:
        demands_map[c] = demands[c]
    
    # Phase 1: Greedy Construction with nearest neighbor
    routes = []
    unassigned = list(customers)
    random.shuffle(unassigned)
    
    for c in unassigned:
        placed = False
        best_idx = -1
        best_pos = -1
        best_new_dist = float('inf')
        best_new_cust = None
        
        for i, r in enumerate(routes):
            if sum(demands_map.get(x, 0) for x in r) + demands_map[c] > capacity:
                continue
            cust_list = list(r)
            for pos in range(len(cust_list) + 1):
                new_cust = cust_list[:pos] + [c] + cust_list[pos:]
                exp, d_exp = _expand_route(new_cust, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
                if exp is not None and d_exp < best_new_dist:
                    best_new_dist = d_exp
                    best_idx = i
                    best_pos = pos
                    best_new_cust = new_cust
        
        if best_idx != -1:
            routes[best_idx] = best_new_cust
            placed = True
        else:
            exp, d_exp = _expand_route([c], dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
            if exp is not None:
                routes.append([c])
                placed = True
    
    if time.time() - start_time > time_limit * 0.3:
        # Not enough time, just do one local search
        routes = _local_search(routes, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations, time_limit, start_time)
        return routes
    
    # Phase 2: ILS with Or-opt perturbation
    best_routes = [list(r) for r in routes]
    best_dist = _total_distance(best_routes, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
    
    current_routes = [list(r) for r in routes]
    current_dist = best_dist
    
    ls_time_budget = time_limit * 0.15
    max_il_s_iters = 200
    
    for iter_i in range(max_il_s_iters):
        if time.time() - start_time > time_limit * 0.85:
            break
        
        # Perturbation: Or-opt move
        n_pert = random.choice([1, 2])
        for _ in range(n_pert):
            if len(current_routes) < 2:
                break
            seg_len = random.choice([1, 2, 3])
            i = random.randint(0, len(current_routes) - 1)
            seq_i = list(current_routes[i])
            if len(seq_i) < seg_len:
                continue
            k = random.randint(0, len(seq_i) - seg_len)
            segment = seq_i[k:k+seg_len]
            new_seq_i = seq_i[:k] + seq_i[k+seg_len:]
            
            # Find a valid route to insert into
            valid_targets = []
            for j in range(len(current_routes)):
                if i == j:
                    continue
                if sum(demands_map.get(x, 0) for x in current_routes[j]) + sum(demands_map.get(x, 0) for x in segment) <= capacity:
                    valid_targets.append(j)
            
            if not valid_targets:
                continue
            j = random.choice(valid_targets)
            seq_j = list(current_routes[j])
            p = random.randint(0, len(seq_j))
            new_seq_j = seq_j[:p] + segment + seq_j[p:]
            
            # Validate
            exp_i, d_i = _expand_route(new_seq_i, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations) if new_seq_i else ([], 0.0)
            exp_j, d_j = _expand_route(new_seq_j, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
            
            if exp_i is None or exp_j is None:
                continue
            
            current_routes[i] = new_seq_i
            current_routes[j] = new_seq_j
        
        # Local search on perturbed solution
        ls_start = time.time()
        current_routes = _local_search(current_routes, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations, ls_time_budget, ls_start)
        
        current_dist = _total_distance(current_routes, dist, ready, due, service, battery, speed, recharge_rate, horizon, stations)
        
        if current_dist < best_dist:
            best_dist = current_dist
            best_routes = [list(r) for r in current_routes]
        else:
            # With some probability, accept worse solutions (simulated annealing style)
            if random.random() < 0.3:
                pass  # Keep current as is for next perturbation
            else:
                current_routes = [list(r) for r in best_routes]
    
    return best_routes

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145
    routes = _solve_instance(instance, time_limit=140)
    return routes
# EVOLVE-BLOCK-END
