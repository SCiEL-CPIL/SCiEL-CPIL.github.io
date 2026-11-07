"""EV routing with time windows and charging - improved algorithm."""
import numpy as np
import math
import time
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n_customers = instance["n_customers"]
    coords = instance["coords"]
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute pairwise distances
    n_nodes = len(coords)
    coords_arr = np.array(coords, dtype=float)
    dist = np.zeros((n_nodes, n_nodes), dtype=float)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords_arr[i][0] - coords_arr[j][0]
            dy = coords_arr[i][1] - coords_arr[j][1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i][j] = d
            dist[j][i] = d

    # Helper: compute total distance of a route
    def route_distance(route):
        if not route:
            return 0.0
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    # Helper: validate a route and return (valid, total_distance)
    def validate_route(route):
        t = 0.0
        bat = battery_cap
        load = 0.0
        prev = 0
        total_dist = 0.0
        for node in route:
            d = dist[prev][node]
            if bat < d - 1e-9:
                return False, 0.0
            t += d / speed
            bat -= d
            load += demands[node]
            if load > capacity + 1e-9:
                return False, 0.0
            total_dist += d
            if 1 <= node <= n_customers:
                if t < ready[node] - 1e-9:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return False, 0.0
                t += service[node]
            elif node in stations:
                deficit = battery_cap - bat
                t += max(0.0, deficit * recharge_rate)
                bat = battery_cap
            prev = node
        d = dist[prev][0]
        if bat < d - 1e-9:
            return False, 0.0
        t += d / speed
        if t > horizon + 1e-9:
            return False, 0.0
        total_dist += d
        return True, total_dist

    # Build a route from a sequence of customers, inserting stations as needed
    def build_route(cust_seq):
        if not cust_seq:
            return []

        route = []
        t = 0.0
        bat = battery_cap
        load = 0.0
        prev = 0

        for cust in cust_seq:
            # Try direct
            d_direct = dist[prev][cust]
            if bat >= d_direct - 1e-9:
                t_arr = t + d_direct / speed
                if t_arr < ready[cust]:
                    t_arr = ready[cust]
                if t_arr > due[cust] + 1e-9:
                    return None
                t = t_arr + service[cust]
                bat -= d_direct
                load += demands[cust]
                if load > capacity + 1e-9:
                    return None
                route.append(cust)
                prev = cust
            else:
                # Need to visit a station
                found = False
                for st in stations:
                    d_to_st = dist[prev][st]
                    if bat < d_to_st - 1e-9:
                        continue
                    t_st = t + d_to_st / speed
                    bat_at_st = bat - d_to_st
                    rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                    t_st += rt
                    bat_st = battery_cap
                    d_st_cust = dist[st][cust]
                    if bat_st < d_st_cust - 1e-9:
                        continue
                    t_cust = t_st + d_st_cust / speed
                    if t_cust < ready[cust]:
                        t_cust = ready[cust]
                    if t_cust > due[cust] + 1e-9:
                        continue
                    t = t_cust + service[cust]
                    bat = bat_st - d_st_cust
                    load += demands[cust]
                    if load > capacity + 1e-9:
                        continue
                    route.append(st)
                    route.append(cust)
                    prev = cust
                    found = True
                    break
                if not found:
                    return None

        # Return to depot
        d_ret = dist[prev][0]
        if bat >= d_ret - 1e-9:
            t_ret = t + d_ret / speed
            if t_ret > horizon + 1e-9:
                return None
        else:
            found = False
            for st in stations:
                d_to_st = dist[prev][st]
                if bat < d_to_st - 1e-9:
                    continue
                t_st = t + d_to_st / speed
                bat_at_st = bat - d_to_st
                rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                t_st += rt
                bat_st = battery_cap
                d_st_dep = dist[st][0]
                if bat_st < d_st_dep - 1e-9:
                    continue
                t_dep = t_st + d_st_dep / speed
                if t_dep > horizon + 1e-9:
                    continue
                route.append(st)
                found = True
                break
            if not found:
                return None

        return route

    # Customer list
    customers = list(range(1, n_customers + 1))

    # Greedy construction with nearest neighbor + time window consideration
    def greedy_construct():
        routes = []
        remaining = set(customers)

        while remaining and time.time() < deadline - 5.0:
            best_route = None
            best_dist = float('inf')
            best_used = None

            # Try starting from each remaining customer
            for start_cust in sorted(remaining, key=lambda v: due[v]):
                route_seq = [start_cust]
                used = {start_cust}
                current_load = demands[start_cust]
                prev_node = start_cust
                t = 0.0
                bat = battery_cap

                # Greedily add the next feasible customer
                while time.time() < deadline - 5.0:
                    best_next = None
                    best_metric = float('inf')

                    for c in remaining - used:
                        if current_load + demands[c] > capacity + 1e-9:
                            continue

                        # Check if we can go from prev_node to c
                        d_direct = dist[prev_node][c]
                        if bat >= d_direct - 1e-9:
                            t_arr = t + d_direct / speed
                            if t_arr < ready[c]:
                                t_arr = ready[c]
                            if t_arr <= due[c] + 1e-9:
                                # Metric: distance + time penalty
                                metric = d_direct + max(0, due[c] - t_arr) * 0.1
                                if metric < best_metric:
                                    best_metric = metric
                                    best_next = c
                        else:
                            # Need station
                            for st in stations:
                                d_to_st = dist[prev_node][st]
                                if bat < d_to_st - 1e-9:
                                    continue
                                t_st = t + d_to_st / speed
                                bat_at_st = bat - d_to_st
                                rt = max(0.0, (battery_cap - bat_at_st) * recharge_rate)
                                t_st += rt
                                bat_st = battery_cap
                                d_st_cust = dist[st][c]
                                if bat_st < d_st_cust - 1e-9:
                                    continue
                                t_cust = t_st + d_st_cust / speed
                                if t_cust < ready[c]:
                                    t_cust = ready[c]
                                if t_cust <= due[c] + 1e-9:
                                    metric = d_to_st + d_st_cust + max(0, due[c] - t_cust) * 0.1
                                    if metric < best_metric:
                                        best_metric = metric
                                        best_next = c
                                break

                    if best_next is None:
                        break

                    route_seq.append(best_next)
                    used.add(best_next)
                    current_load += demands[best_next]
                    prev_node = best_next

                # Finalize the route
                final_route = build_route(route_seq)
                if final_route is not None:
                    valid, d = validate_route(final_route)
                    if valid and d < best_dist:
                        best_dist = d
                        best_route = final_route
                        best_used = set(c for c in route_seq if 1 <= c <= n_customers)

            if best_route is None:
                # Fallback: single customer route
                for c in remaining:
                    single_route = build_route([c])
                    if single_route is not None:
                        valid, d = validate_route(single_route)
                        if valid:
                            best_route = single_route
                            best_used = {c}
                            break
                if best_route is None:
                    c = list(remaining)[0]
                    best_route = [c]
                    best_used = {c}

            routes.append(best_route)
            remaining -= best_used

        # Add any remaining customers as single routes
        for c in remaining:
            single_route = build_route([c])
            if single_route is not None:
                routes.append(single_route)
            else:
                routes.append([c])

        return routes

    # Local search: relocate customers between routes
    def local_search_relocate(routes, max_iter=20):
        improved = True
        iteration = 0
        while improved and iteration < max_iter and time.time() < deadline - 5.0:
            improved = False
            iteration += 1

            for i in range(len(routes)):
                if time.time() > deadline - 5.0:
                    break
                for j in range(len(routes[i])):
                    cust = routes[i][j]
                    if not (1 <= cust <= n_customers):
                        continue

                    for k in range(len(routes)):
                        if k == i or time.time() > deadline - 5.0:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            new_route_i = routes[i][:j] + routes[i][j + 1:]
                            new_route_k = routes[k][:pos] + [cust] + routes[k][pos:]

                            r_i = build_route(new_route_i)
                            r_k = build_route(new_route_k)

                            if r_i is None or r_k is None:
                                continue
                            valid_i, dist_i = validate_route(r_i)
                            valid_k, dist_k = validate_route(r_k)
                            if not valid_i or not valid_k:
                                continue

                            old_dist = route_distance(routes[i]) + route_distance(routes[k])
                            new_dist = dist_i + dist_k

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

        return routes

    # 2-opt within each route
    def two_opt_route(route):
        cust_seq = [c for c in route if 1 <= c <= n_customers]
        if len(cust_seq) < 2:
            return route

        best_seq = cust_seq[:]
        best_route = build_route(cust_seq)
        if best_route is None:
            return route
        best_dist, _ = validate_route(best_route)

        for a in range(len(best_seq)):
            for b in range(a + 1, len(best_seq)):
                if time.time() > deadline - 5.0:
                    break
                new_seq = best_seq[:a] + best_seq[a:b + 1][::-1] + best_seq[b + 1:]
                new_route = build_route(new_seq)
                if new_route is not None:
                    valid, d = validate_route(new_route)
                    if valid and d < best_dist - 1e-6:
                        best_dist = d
                        best_seq = new_seq[:]
                        best_route = new_route

        if best_route is not None:
            return best_route
        return route

    # Main algorithm
    # 1. Greedy construction
    routes = greedy_construct()

    # 2. Local search: relocate
    if time.time() < deadline - 10.0:
        routes = local_search_relocate(routes, max_iter=30)

    # 3. 2-opt within each route
    if time.time() < deadline - 5.0:
        for i in range(len(routes)):
            if time.time() > deadline - 5.0:
                break
            routes[i] = two_opt_route(routes[i])

    # 4. Another round of relocate after 2-opt
    if time.time() < deadline - 5.0:
        routes = local_search_relocate(routes, max_iter=15)

    # Final validation
    valid_routes = []
    covered = set()
    for r in routes:
        valid, d = validate_route(r)
        if valid:
            valid_routes.append(r)
            for node in r:
                if 1 <= node <= n_customers:
                    covered.add(node)

    # Add any missing customers
    for cust in range(1, n_customers + 1):
        if cust not in covered:
            single_route = build_route([cust])
            if single_route is not None:
                valid_routes.append(single_route)
            else:
                valid_routes.append([cust])
            covered.add(cust)

    return valid_routes
