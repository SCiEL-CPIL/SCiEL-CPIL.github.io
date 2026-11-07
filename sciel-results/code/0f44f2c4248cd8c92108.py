"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import heapq

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

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

    # Precompute pairwise distances using NumPy vectorization
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    diff = coords_arr[:, np.newaxis, :] - coords_arr[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    dist_list = dist.tolist()  # Faster for scalar access

    station_set = set(stations)

    # Fast route simulation: returns (total_dist, valid) or (inf, False)
    # route is a list of node ids (customers and stations)
    def simulate_route(route):
        """Simulate route, return (total_distance, is_valid)."""
        t = 0.0
        bat = float(battery_cap)
        load = 0
        total_d = 0.0
        prev = 0
        for node in route:
            d = dist_list[prev][node]
            if bat < d - 1e-9:
                return total_d, False
            t += d
            bat -= d
            total_d += d
            load += demands[node]
            if load > capacity:
                return total_d, False
            if 1 <= node <= n_customers:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return total_d, False
                t += service[node]
            elif node in station_set:
                deficit = battery_cap - bat
                t += max(0.0, deficit * recharge_rate)
                bat = float(battery_cap)
            prev = node
        d = dist_list[prev][0]
        if bat < d - 1e-9:
            return total_d, False
        t += d
        total_d += d
        if t > horizon + 1e-9:
            return total_d, False
        return total_d, True

    # Build a route given a sequence of customer visits (with stations inserted as needed)
    def build_route_from_sequence(cust_seq):
        """
        Given a sequence of customers, build a valid route by inserting stations as needed.
        Returns (route, total_dist) or (None, inf) if infeasible.
        """
        if not cust_seq:
            return [], 0.0
        
        # Quick capacity check
        total_load = 0
        for c in cust_seq:
            total_load += demands[c]
        if total_load > capacity:
            return None, float('inf')
        
        route = []
        t = 0.0
        bat = float(battery_cap)
        load = 0
        prev = 0
        total_d = 0.0
        
        for cust in cust_seq:
            d_direct = dist_list[prev][cust]
            if bat >= d_direct - 1e-9:
                # Can go directly
                t_arr = t + d_direct
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return None, float('inf')
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                route.append(cust)
                prev = cust
                total_d += d_direct
            else:
                # Need to visit a station first
                found = False
                for st in stations:
                    d_to_st = dist_list[prev][st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st
                    bat_at_st = bat - d_to_st
                    deficit = battery_cap - bat_at_st
                    recharge_time = max(0.0, deficit * recharge_rate)
                    t_st += recharge_time
                    bat_st = float(battery_cap)
                    d_st_to_cust = dist_list[st][cust]
                    if bat_st < d_st_to_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_to_cust
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    t = t_cust + service[cust]
                    bat = bat_st - d_st_to_cust
                    load += demands[cust]
                    route.append(st)
                    route.append(cust)
                    prev = cust
                    total_d += d_to_st + d_st_to_cust
                    found = True
                    break
                if not found:
                    return None, float('inf')
        
        # Check return to depot
        d_return = dist_list[prev][0]
        if bat < d_return - 1e-9:
            # Try to go through a station
            found = False
            for st in stations:
                d_to_st = dist_list[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st
                bat_at_st = bat - d_to_st
                deficit = battery_cap - bat_at_st
                recharge_time = max(0.0, deficit * recharge_rate)
                t_st += recharge_time
                bat_st = float(battery_cap)
                d_st_to_depot = dist_list[st][0]
                if bat_st < d_st_to_depot - 1e-9:
                    continue
                t_depot = t_st + d_st_to_depot
                if t_depot > horizon + 1e-9:
                    continue
                route.append(st)
                total_d += d_to_st + d_st_to_depot
                found = True
                break
            if not found:
                return None, float('inf')
        else:
            t_return = t + d_return
            if t_return > horizon + 1e-9:
                return None, float('inf')
            total_d += d_return
        
        return route, total_d

    # Savings-based construction
    customers = list(range(1, n_customers + 1))
    
    # Compute savings for each pair of customers
    # savings[i][j] = dist[0][i] + dist[0][j] - dist[i][j]
    savings = {}
    for i in customers:
        for j in customers:
            if i != j:
                s = dist_list[0][i] + dist_list[0][j] - dist_list[i][j]
                savings[(i, j)] = s
    
    # Sort savings in descending order
    sorted_savings = sorted(savings.items(), key=lambda x: -x[1])
    
    # Initialize: each customer is its own route
    # routes_map: customer_id -> (route_list, load, total_dist)
    # Use a dict to track which customer belongs to which route
    customer_route = {c: [c] for c in customers}
    route_load = {c: demands[c] for c in customers}
    route_dist = {c: 0.0 for c in customers}
    # Track route membership: route_id -> set of customers
    route_members = {c: {c} for c in customers}
    # For each route, track the sequence
    route_seq = {c: [c] for c in customers}
    
    # Union-Find-like structure
    parent = {c: c for c in customers}
    
    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    
    def union(c1, c2):
        r1, r2 = find(c1), find(c2)
        if r1 == r2:
            return
        # Merge r2 into r1 (r1 has smaller id for consistency)
        if r1 > r2:
            r1, r2 = r2, r1
        parent[r2] = r1
    
    # Process savings in order
    for (i, j), s in sorted_savings:
        if time.time() > deadline - 30.0:
            break
        r_i, r_j = find(i), find(j)
        if r_i == r_j:
            continue
        # Check capacity
        if route_load[r_i] + route_load[r_j] > capacity:
            continue
        # Try merging: insert j's route after i in i's route
        # We need to check if the merged route is feasible
        seq_i = route_seq[r_i][:]
        seq_j = route_seq[r_j][:]
        
        # Try inserting seq_j at the end of seq_i (after i)
        merged_seq = seq_i + seq_j
        merged_route, merged_dist = build_route_from_sequence(merged_seq)
        if merged_route is None:
            # Try reversed
            merged_seq = seq_i + seq_j[::-1]
            merged_route, merged_dist = build_route_from_sequence(merged_seq)
        
        if merged_route is not None:
            # Perform the merge
            union(i, j)
            new_root = find(i)
            route_seq[new_root] = [c for c in merged_seq if 1 <= c <= n_customers]
            # Rebuild to get correct sequence
            # Actually, let's just track the customer sequence
            cust_merged = [c for c in merged_seq if 1 <= c <= n_customers]
            route_seq[new_root] = cust_merged
            route_load[new_root] = route_load[r_i] + route_load[r_j]
            route_dist[new_root] = merged_dist
            # Clear old routes
            route_seq[r_i] = []
            route_seq[r_j] = []
            route_load[r_i] = 0
            route_load[r_j] = 0
    
    # Extract final routes
    routes = []
    seen_roots = set()
    for c in customers:
        root = find(c)
        if root not in seen_roots and route_seq[root]:
            seen_roots.add(root)
            seq = route_seq[root]
            r, d = build_route_from_sequence(seq)
            if r is not None:
                routes.append(r)
            else:
                # Fallback: split into singles
                for cc in seq:
                    sr, sd = build_route_from_sequence([cc])
                    if sr is not None:
                        routes.append(sr)
                    else:
                        routes.append([cc])
    
    # Verify all customers are covered
    covered = set()
    for r in routes:
        for node in r:
            if 1 <= node <= n_customers:
                covered.add(node)
    
    for c in customers:
        if c not in covered:
            sr, sd = build_route_from_sequence([c])
            if sr is not None:
                routes.append(sr)
            else:
                routes.append([c])
    
    # Local search: relocate and 2-opt
    if time.time() < deadline - 15.0:
        # Precompute route distances
        route_dists = []
        for r in routes:
            d, valid = simulate_route(r)
            route_dists.append(d)
        
        # Relocate: move a customer from one route to another
        improved = True
        max_iter = 30
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline - 10.0:
            improved = False
            iteration += 1
            
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                if not routes[i]:
                    continue
                # Get customer sequence for route i
                cust_seq_i = [c for c in routes[i] if 1 <= c <= n_customers]
                if len(cust_seq_i) < 2:
                    continue
                
                for pos in range(len(cust_seq_i)):
                    if time.time() > deadline - 10.0:
                        break
                    cust = cust_seq_i[pos]
                    
                    # Remove cust from route i
                    new_seq_i = cust_seq_i[:pos] + cust_seq_i[pos+1:]
                    if not new_seq_i:
                        r_i, d_i = [], 0.0
                    else:
                        r_i, d_i = build_route_from_sequence(new_seq_i)
                        if r_i is None:
                            continue
                    
                    # Try inserting into each other route
                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 10.0:
                            continue
                        cust_seq_k = [c for c in routes[k] if 1 <= c <= n_customers]
                        # Check capacity
                        load_k = sum(demands[c] for c in cust_seq_k)
                        if load_k + demands[cust] > capacity:
                            continue
                        
                        for ins_pos in range(len(cust_seq_k) + 1):
                            if time.time() > deadline - 10.0:
                                break
                            new_seq_k = cust_seq_k[:ins_pos] + [cust] + cust_seq_k[ins_pos:]
                            r_k, d_k = build_route_from_sequence(new_seq_k)
                            if r_k is None:
                                continue
                            
                            new_total = d_i + d_k
                            old_total = route_dists[i] + route_dists[k]
                            
                            if new_total < old_total - 1e-6:
                                routes[i] = r_i
                                routes[k] = r_k
                                route_dists[i] = d_i
                                route_dists[k] = d_k
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # 2-opt within each route
        if time.time() < deadline - 10.0:
            for i in range(len(routes)):
                if time.time() > deadline - 10.0:
                    break
                cust_seq = [c for c in routes[i] if 1 <= c <= n_customers]
                if len(cust_seq) < 2:
                    continue
                
                best_seq = cust_seq[:]
                best_dist, _ = build_route_from_sequence(best_seq)
                if best_dist == float('inf'):
                    continue
                
                n = len(best_seq)
                for a in range(n):
                    if time.time() > deadline - 10.0:
                        break
                    for b in range(a+1, n):
                        if time.time() > deadline - 10.0:
                            break
                        new_seq = best_seq[:a] + best_seq[a:b+1][::-1] + best_seq[b+1:]
                        r, d = build_route_from_sequence(new_seq)
                        if r is not None and d < best_dist - 1e-6:
                            best_dist = d
                            best_seq = new_seq[:]
                
                if best_seq != cust_seq:
                    r, d = build_route_from_sequence(best_seq)
                    if r is not None:
                        routes[i] = r
                        route_dists[i] = d
    
    # Final validation and cleanup
    valid_routes = []
    covered = set()
    for r in routes:
        d, valid = simulate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)
    
    # Add any missing customers
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            r, d = build_route_from_sequence([cust])
            if r is not None:
                valid_routes.append(r)
            else:
                valid_routes.append([cust])
            covered.add(cust)
    
    return valid_routes
# EVOLVE-BLOCK-END
