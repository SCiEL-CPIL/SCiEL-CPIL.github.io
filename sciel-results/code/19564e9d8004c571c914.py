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

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d

            if bat < -1e-9:
                return False, 0.0

            # At customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d

        if bat < -1e-9:
            return False, 0.0
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

    def fast_check_route(route):
        """Faster feasibility check with early termination."""
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route)
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

    def compute_route_profile(route):
        """Compute arrival times and battery levels at each node in route.
        Returns (arrival_times, battery_levels, feasible) where arrays are indexed by position."""
        n = len(route)
        if n == 0:
            return np.array([]), np.array([]), True

        arr_t = np.zeros(n)
        arr_b = np.zeros(n)
        arr_b[0] = battery - dist[0, route[0]]
        if arr_b[0] < -1e-9:
            return arr_t, arr_b, False
        arr_t[0] = dist[0, route[0]] / speed
        if arr_t[0] < ready[route[0]]:
            arr_t[0] = ready[route[0]]
        if arr_t[0] > due[route[0]] + 1e-9:
            return arr_t, arr_b, False
        arr_t[0] += service[route[0]]

        for i in range(1, n):
            d = dist[route[i-1], route[i]]
            arr_b[i] = arr_b[i-1] - d
            if arr_b[i] < -1e-9:
                return arr_t, arr_b, False
            arr_t[i] = arr_t[i-1] + d / speed
            if arr_t[i] < ready[route[i]]:
                arr_t[i] = ready[route[i]]
            if arr_t[i] > due[route[i]] + 1e-9:
                return arr_t, arr_b, False
            arr_t[i] += service[route[i]]

        # Check return to depot
        d_ret = dist[route[-1], 0]
        final_bat = arr_b[-1] - d_ret
        final_t = arr_t[-1] + d_ret / speed
        if final_bat < -1e-9 or final_t > horizon + 1e-9:
            return arr_t, arr_b, False

        return arr_t, arr_b, True

    def quick_feasible_insert(route, pos, v, arr_t, arr_b):
        """Quick check if inserting v at position pos in route is feasible,
        given precomputed arrival times and battery levels."""
        n = len(route)
        if pos == 0:
            # Insert at beginning
            d_in = dist[0, v]
            bat_after_in = battery - d_in
            if bat_after_in < -1e-9:
                return False
            t_in = d_in / speed
            if t_in < ready[v]:
                t_in = ready[v]
            if t_in > due[v] + 1e-9:
                return False
            t_in += service[v]
            # Check transition to first customer
            if n > 0:
                d_out = dist[v, route[0]]
                bat_after_out = bat_after_in - d_out
                if bat_after_out < -1e-9:
                    return False
                t_out = t_in + d_out / speed
                if t_out < ready[route[0]]:
                    t_out = ready[route[0]]
                if t_out > due[route[0]] + 1e-9:
                    return False
                # Rest of route unchanged in terms of relative timing, but absolute time shifted
                # Actually we need to check that the shift doesn't violate any time windows
                # The shift is: new_time_at_i = old_time_at_i + (t_in + d_out/speed) - arr_t[0]
                # But this is complex; for now just do full check if n > 5
                if n > 5:
                    new_route = [v] + route
                    return fast_check_route(new_route)[0]
            # Check return
            d_ret = dist[v, 0] if n == 0 else dist[route[-1], 0]
            # Actually if inserting at start, return path unchanged
            return True
        elif pos == n:
            # Insert at end
            d_in = dist[route[-1], v]
            bat_after_in = arr_b[-1] - d_in
            if bat_after_in < -1e-9:
                return False
            t_in = arr_t[-1] + d_in / speed
            if t_in < ready[v]:
                t_in = ready[v]
            if t_in > due[v] + 1e-9:
                return False
            t_in += service[v]
            # Check return
            d_ret = dist[v, 0]
            bat_ret = bat_after_in - d_ret
            t_ret = t_in + d_ret / speed
            if bat_ret < -1e-9 or t_ret > horizon + 1e-9:
                return False
            return True
        else:
            # Insert in middle - more complex, do full check for safety
            new_route = route[:pos] + [v] + route[pos:]
            return fast_check_route(new_route)[0]

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
        """Construct routes using greedy nearest neighbor with time windows."""
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

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    def relocate_search(routes, max_passes=8):
        """Remove-and-reinsert local search."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    v = routes[i][j]
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = fast_check_route(route_i_without)
                        if not feasible_i:
                            continue
                    else:
                        feasible_i = True

                    for k in range(len(new_routes)):
                        if time.time() > deadline:
                            break
                        route_k = new_routes[k]
                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [v] + route_k[pos:]
                            feasible, _ = fast_check_route(new_route_k)
                            if not feasible:
                                continue

                            new_routes[k] = new_route_k
                            new_dist = total_route_distance(new_routes)
                            if new_dist < current_dist - 1e-6:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                            else:
                                new_routes[k] = route_k
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # Cross-route swap local search
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
                            new_routes = []
                            for r in range(len(best_routes)):
                                if r == i:
                                    nr = best_routes[i][:]
                                    nr[a] = vj
                                    new_routes.append(nr)
                                elif r == j:
                                    nr = best_routes[j][:]
                                    nr[b] = vi
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(best_routes[r][:])

                            feasible = True
                            for route in new_routes:
                                f, _ = fast_check_route(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    best_routes = new_routes
                                    current_dist = new_dist
                                    improved_swap = True
                                    break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish: 2-opt + relocate
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=3)

        # Final 2-opt
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
