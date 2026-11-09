"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix with NumPy
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def fast_check_route(route):
        """Faster feasibility check with early termination. Supports station nodes."""
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route if v not in station_set)
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            t += d / speed
            total_dist += d

            if v in station_set:
                # Recharge: time = (battery - energy) * recharge_rate, then refill
                charge_time = (bat) * recharge_rate
                t += charge_time
                bat = battery
            else:
                if t < ready[v]:
                    t = ready[v]
                if t > due[v] + 1e-9:
                    return False, 0.0
                t += service[v]
            prev = v

        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0
        t += d / speed
        total_dist += d

        if t > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def total_route_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
            total += dist[route[-1], 0]
        return total

    def savings_construct():
        """Clarke-Wright savings based construction with feasibility checks."""
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0:
                    savings.append((s, vi, vj))

        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of.get(vi)
            rj = route_of.get(vj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            can_merge = False
            new_route = None

            # Try all four merge options
            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append(route_i + route_j)
            if route_j[-1] == vj and route_i[0] == vi:
                options.append(route_j + route_i)
            if route_i[0] == vi and route_j[0] == vj:
                options.append(route_i[::-1] + route_j)
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append(route_i + route_j[::-1])
            if route_i[0] == vi and route_j[-1] == vj:
                options.append(route_i[::-1] + route_j[::-1])
            if route_j[0] == vj and route_i[-1] == vi:
                options.append(route_j + route_i[::-1])

            for candidate in options:
                if sum(demands[x] for x in candidate) > capacity:
                    continue
                f, _ = fast_check_route(candidate)
                if f:
                    can_merge = True
                    new_route = candidate
                    break

            if can_merge:
                routes[ri] = new_route
                routes[rj] = []
                for v in new_route:
                    route_of[v] = ri
                # Clean up empty routes
                non_empty = [r for r in routes if r]
                for i, r in enumerate(non_empty):
                    for v in r:
                        route_of[v] = i
                routes = non_empty

        return [r for r in routes if r]

    def greedy_construct(customer_order):
        """Construct routes using greedy nearest neighbor with time windows and charging."""
        remaining = list(customer_order)
        routes = []

        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0

            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1

                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed

                    if bat_after < -1e-9:
                        # Try going through a station first
                        for st in stations:
                            d1 = dist[prev, st]
                            d2 = dist[st, v]
                            bat_st = bat - d1
                            if bat_st < -1e-9:
                                continue
                            t_st = t + d1 / speed
                            charge_time = bat_st * recharge_rate
                            t_st += charge_time
                            t_v = t_st + d2 / speed
                            bat_v = battery - d2
                            if bat_v < -1e-9:
                                continue
                            if load + demands[v] > capacity:
                                continue
                            if t_v < ready[v]:
                                t_v = ready[v]
                            if t_v > due[v] + 1e-9:
                                continue
                            time_slack = due[v] - t_v
                            score = d1 + d2 - 0.02 * time_slack
                            if score < best_score:
                                best_score = score
                                best_customer = v
                                best_idx = idx
                        continue

                    if load + demands[v] > capacity:
                        continue

                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue

                    # Score: prioritize distance, then time tightness
                    time_slack = due[v] - t_after
                    score = d - 0.02 * time_slack

                    if score < best_score:
                        best_score = score
                        best_customer = v
                        best_idx = idx

                if best_customer is None:
                    break

                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]

                if t < ready[v]:
                    t = ready[v]
                t += service[v]

                route.append(v)
                remaining.pop(best_idx)
                prev = v

            if route:
                routes.append(route)

        return routes

    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
        all_customers = []
        for route in routes:
            feasible, _ = fast_check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    # Initial solution: one customer per route (always feasible)
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try savings construction
    if time.time() < deadline:
        sw_routes = savings_construct()
        if validate_routes(sw_routes):
            d = total_route_distance(sw_routes)
            if d < best_distance:
                best_routes = sw_routes
                best_distance = d

    # Try greedy construction with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = greedy_routes
                best_distance = d

    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement."""
        if len(route) < 2:
            return route
        improved = True
        while improved:
            improved = False
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    old_boundary = 0.0
                    if a > 0:
                        old_boundary += dist[route[a - 1], route[a]]
                    else:
                        old_boundary += dist[0, route[a]]
                    if b < n - 1:
                        old_boundary += dist[route[b], route[b + 1]]
                    else:
                        old_boundary += dist[route[b], 0]

                    new_boundary = 0.0
                    if a > 0:
                        new_boundary += dist[route[a - 1], route[b]]
                    else:
                        new_boundary += dist[0, route[b]]
                    if b < n - 1:
                        new_boundary += dist[route[a], route[b + 1]]
                    else:
                        new_boundary += dist[route[a], 0]

                    if new_boundary < old_boundary - 1e-6:
                        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        feasible, _ = fast_check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def relocate_search_fast(routes, max_passes=8):
        """Remove-and-reinsert local search with Or-opt (1-3 consecutive nodes) using delta distance."""
        routes = [list(r) for r in routes]

        def compute_route_dist(route):
            if not route:
                return 0.0
            d = dist[0, route[0]]
            for i in range(len(route) - 1):
                d += dist[route[i], route[i + 1]]
            d += dist[route[-1], 0]
            return d

        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = sum(compute_route_dist(r) for r in routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for seg_len in [1, 2, 3]:
                    if time.time() > deadline:
                        break
                    for j in range(len(routes[i]) - seg_len + 1):
                        if time.time() > deadline:
                            break
                        segment = routes[i][j:j + seg_len]
                        route_i_without = routes[i][:j] + routes[i][j + seg_len:]

                        if not route_i_without:
                            continue  # Don't remove entire route

                        # Check feasibility of source route without segment
                        feasible_i, _ = fast_check_route(route_i_without)
                        if not feasible_i:
                            continue

                        # Compute delta distance for removal
                        old_src_dist = compute_route_dist(routes[i])
                        new_src_dist = compute_route_dist(route_i_without)

                        for k in range(len(routes)):
                            if time.time() > deadline:
                                break
                            if k == i:
                                continue
                            route_k = routes[k]
                            for pos in range(len(route_k) + 1):
                                new_route_k = route_k[:pos] + segment + route_k[pos:]
                                feasible, _ = fast_check_route(new_route_k)
                                if not feasible:
                                    continue

                                old_dst_dist = compute_route_dist(route_k)
                                new_dst_dist = compute_route_dist(new_route_k)

                                delta = (new_src_dist + new_dst_dist) - (old_src_dist + old_dst_dist)
                                if delta < -1e-6:
                                    routes[i] = route_i_without
                                    routes[k] = new_route_k
                                    current_dist += delta
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    def relocate_search(routes, max_passes=8):
        """Remove-and-reinsert local search with Or-opt (1-3 consecutive nodes)."""
        return relocate_search_fast(routes, max_passes)

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=8)

    # Cross-route swap local search (only check 2 modified routes)
    if time.time() < deadline:
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            current_dist = total_route_distance(best_routes)
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_routes)):
                    if time.time() > deadline:
                        continue
                    for a in range(len(best_routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(best_routes[j])):
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            if vi == vj:
                                continue
                            # Compute distance delta without full reconstruction
                            old_d = 0.0
                            if a > 0:
                                old_d += dist[best_routes[i][a-1], vi]
                            else:
                                old_d += dist[0, vi]
                            if a < len(best_routes[i]) - 1:
                                old_d += dist[vi, best_routes[i][a+1]]
                            else:
                                old_d += dist[vi, 0]
                            if b > 0:
                                old_d += dist[best_routes[j][b-1], vj]
                            else:
                                old_d += dist[0, vj]
                            if b < len(best_routes[j]) - 1:
                                old_d += dist[vj, best_routes[j][b+1]]
                            else:
                                old_d += dist[vj, 0]

                            new_d = 0.0
                            if a > 0:
                                new_d += dist[best_routes[i][a-1], vj]
                            else:
                                new_d += dist[0, vj]
                            if a < len(best_routes[i]) - 1:
                                new_d += dist[vj, best_routes[i][a+1]]
                            else:
                                new_d += dist[vj, 0]
                            if b > 0:
                                new_d += dist[best_routes[j][b-1], vi]
                            else:
                                new_d += dist[0, vi]
                            if b < len(best_routes[j]) - 1:
                                new_d += dist[vi, best_routes[j][b+1]]
                            else:
                                new_d += dist[vi, 0]

                            if new_d >= old_d - 1e-6:
                                continue

                            new_routes = [r[:] for r in best_routes]
                            new_routes[i][a] = vj
                            new_routes[j][b] = vi

                            f1, _ = fast_check_route(new_routes[i])
                            if not f1:
                                continue
                            f2, _ = fast_check_route(new_routes[j])
                            if not f2:
                                continue

                            best_routes = new_routes
                            current_dist = total_route_distance(best_routes)
                            improved_swap = True
                            break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish: multiple rounds of 2-opt + relocate + swap
    polish_rounds = 4
    for round_num in range(polish_rounds):
        if time.time() > deadline:
            break

        # 2-opt on each route
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        # Relocate search
        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=6)

        # Cross-route swap
        if time.time() < deadline:
            improved_swap = True
            while improved_swap and time.time() < deadline:
                improved_swap = False
                current_dist = total_route_distance(best_routes)
                for i in range(len(best_routes)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(best_routes)):
                        if time.time() > deadline:
                            continue
                        for a in range(len(best_routes[i])):
                            if time.time() > deadline:
                                break
                            for b in range(len(best_routes[j])):
                                vi = best_routes[i][a]
                                vj = best_routes[j][b]
                                if vi == vj:
                                    continue
                                old_d = 0.0
                                if a > 0:
                                    old_d += dist[best_routes[i][a-1], vi]
                                else:
                                    old_d += dist[0, vi]
                                if a < len(best_routes[i]) - 1:
                                    old_d += dist[vi, best_routes[i][a+1]]
                                else:
                                    old_d += dist[vi, 0]
                                if b > 0:
                                    old_d += dist[best_routes[j][b-1], vj]
                                else:
                                    old_d += dist[0, vj]
                                if b < len(best_routes[j]) - 1:
                                    old_d += dist[vj, best_routes[j][b+1]]
                                else:
                                    old_d += dist[vj, 0]

                                new_d = 0.0
                                if a > 0:
                                    new_d += dist[best_routes[i][a-1], vj]
                                else:
                                    new_d += dist[0, vj]
                                if a < len(best_routes[i]) - 1:
                                    new_d += dist[vj, best_routes[i][a+1]]
                                else:
                                    new_d += dist[vj, 0]
                                if b > 0:
                                    new_d += dist[best_routes[j][b-1], vi]
                                else:
                                    new_d += dist[0, vi]
                                if b < len(best_routes[j]) - 1:
                                    new_d += dist[vi, best_routes[j][b+1]]
                                else:
                                    new_d += dist[vi, 0]

                                if new_d >= old_d - 1e-6:
                                    continue

                                new_routes = [r[:] for r in best_routes]
                                new_routes[i][a] = vj
                                new_routes[j][b] = vi

                                f1, _ = fast_check_route(new_routes[i])
                                if not f1:
                                    continue
                                f2, _ = fast_check_route(new_routes[j])
                                if not f2:
                                    continue

                                best_routes = new_routes
                                current_dist = total_route_distance(best_routes)
                                improved_swap = True
                                break
                            if improved_swap:
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

        # Final 2-opt after swap
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # High-precision final polish: tight tolerance, extra 2-opt passes
    if time.time() < deadline - 5.0:
        for i in range(len(best_routes)):
            if time.time() > deadline - 2.0:
                break
            route = best_routes[i]
            improved_hp = True
            while improved_hp and time.time() < deadline - 2.0:
                improved_hp = False
                n = len(route)
                for a in range(n - 1):
                    if time.time() > deadline - 2.0:
                        break
                    for b in range(a + 1, n):
                        old_boundary = 0.0
                        if a > 0:
                            old_boundary += dist[route[a - 1], route[a]]
                        else:
                            old_boundary += dist[0, route[a]]
                        if b < n - 1:
                            old_boundary += dist[route[b], route[b + 1]]
                        else:
                            old_boundary += dist[route[b], 0]

                        new_boundary = 0.0
                        if a > 0:
                            new_boundary += dist[route[a - 1], route[b]]
                        else:
                            new_boundary += dist[0, route[b]]
                        if b < n - 1:
                            new_boundary += dist[route[a], route[b + 1]]
                        else:
                            new_boundary += dist[route[a], 0]

                        if new_boundary < old_boundary - 1e-8:
                            new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                            feasible, _ = fast_check_route(new_route)
                            if feasible:
                                route = new_route
                                improved_hp = True
                                break
                    if improved_hp:
                        break
            best_routes[i] = route

        # One more relocate pass with tighter settings
        if time.time() < deadline - 2.0:
            best_routes = relocate_search(best_routes, max_passes=3)

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
