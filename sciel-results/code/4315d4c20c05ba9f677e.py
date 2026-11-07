# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    DEADLINE = 140.0

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    station_set = set(stations)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    customers = list(range(1, n_customers + 1))

    # Distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Precompute: for each node, can we reach each station, and what's the cost?
    # Also precompute: which customers are reachable from each node directly or via station
    # This is for fast feasibility during local search

    def time_limit_exceeded():
        return time.time() - start_time > DEADLINE

    def simulate(route):
        """Simulate a route. Returns (valid, total_dist, end_time, end_battery)."""
        if not route:
            return True, 0.0, 0.0, battery_max
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        total_demand = 0.0

        for node in route:
            d = dist[curr_node][node]
            total_dist += d
            if curr_battery < d - 1e-9:
                return False, 0.0, 0.0, 0.0
            curr_battery -= d
            curr_time += d / speed

            if node in station_set:
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
            else:
                total_demand += demands[node]
                if total_demand > capacity + 1e-9:
                    return False, 0.0, 0.0, 0.0
                service_start = max(curr_time, ready[node])
                if service_start > due[node] + 1e-9:
                    return False, 0.0, 0.0, 0.0
                curr_time = service_start + service[node]
            curr_node = node

        d = dist[curr_node][0]
        total_dist += d
        if curr_battery < d - 1e-9:
            return False, 0.0, 0.0, 0.0
        curr_time += d / speed
        if curr_time > horizon + 1e-9:
            return False, 0.0, 0.0, 0.0

        return True, total_dist, curr_time, curr_battery

    def is_feasible(route):
        valid, _, _, _ = simulate(route)
        return valid

    def route_dist(route):
        if not route:
            return 0.0
        curr = 0
        total = 0.0
        for node in route:
            total += dist[curr][node]
            curr = node
        total += dist[curr][0]
        return total

    def total_dist(routes):
        return sum(route_dist(r) for r in routes)

    def validate_solution(routes):
        visited = set()
        for route in routes:
            if not is_feasible(route):
                return False
            for node in route:
                if node not in station_set:
                    if node in visited:
                        return False
                    visited.add(node)
        return visited == set(customers)

    def construct_route(ordered_customers):
        """Build a feasible route from an ordered list of customers, inserting stations as needed."""
        route = []
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_demand = 0.0

        for c in ordered_customers:
            d_direct = dist[curr_node][c]
            if curr_battery >= d_direct - 1e-9:
                new_time = curr_time + d_direct / speed
                service_start = max(new_time, ready[c])
                if service_start <= due[c] + 1e-9:
                    route.append(c)
                    curr_battery -= d_direct
                    curr_time = service_start + service[c]
                    total_demand += demands[c]
                    curr_node = c
                    continue

            # Need to charge
            best_station = None
            best_total_dist = float('inf')
            for s in stations:
                d1 = dist[curr_node][s]
                if curr_battery < d1 - 1e-9:
                    continue
                charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                new_time = curr_time + d1 / speed + charge_time
                d2 = dist[s][c]
                new_time2 = new_time + d2 / speed
                service_start = max(new_time2, ready[c])
                if service_start <= due[c] + 1e-9:
                    total_d = d1 + d2
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_station = s

            if best_station is None:
                return route

            s = best_station
            d1 = dist[curr_node][s]
            curr_battery -= d1
            curr_time += d1 / speed
            charge_time = (battery_max - curr_battery) * recharge_rate
            curr_time += charge_time
            curr_battery = battery_max
            route.append(s)
            curr_node = s

            d2 = dist[curr_node][c]
            curr_battery -= d2
            curr_time += d2 / speed
            service_start = max(curr_time, ready[c])
            curr_time = service_start + service[c]
            total_demand += demands[c]
            curr_node = c

        return route

    def greedy_insertion(sort_key="due"):
        """Greedy insertion: insert each customer at the cheapest feasible position."""
        routes = []
        if sort_key == "due":
            sorted_customers = sorted(customers, key=lambda v: due[v])
        elif sort_key == "angle":
            depot_x, depot_y = coords[0][0], coords[0][1]
            sorted_customers = sorted(customers, key=lambda v: math.atan2(coords[v][1] - depot_y, coords[v][0] - depot_x))
        elif sort_key == "nearest":
            sorted_customers = sorted(customers, key=lambda v: dist[0][v])
        else:
            sorted_customers = customers[:]

        for c in sorted_customers:
            if time_limit_exceeded():
                break
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')

            for r_idx, route in enumerate(routes):
                current_demand = sum(demands[node] for node in route if node not in station_set)
                if current_demand + demands[c] > capacity + 1e-9:
                    continue
                for p in range(len(route) + 1):
                    new_route = route[:p] + [c] + route[p:]
                    valid, actual_dist = simulate(new_route)
                    if valid and actual_dist < best_cost:
                        best_cost = actual_dist
                        best_route_idx = r_idx
                        best_pos = p

            if best_route_idx != -1:
                routes[best_route_idx].insert(best_pos, c)
            else:
                routes.append([c])

        return routes

    def clarke_wright_savings():
        """Clarke-Wright savings heuristic with feasibility checks."""
        # Start with each customer as its own route
        routes = [[c] for c in customers]
        route_of = {c: i for i, c in enumerate(customers)}
        # Precompute savings: saving(i,j) = dist(0,i) + dist(0,j) - dist(i,j)
        # But we need to respect constraints, so we'll do iterative merging

        # Sort customers by due date for initial ordering within potential merges
        sorted_custs = sorted(customers, key=lambda v: due[v])

        # Compute savings between adjacent-in-savings customers
        # Standard CW: sort by savings descending, try to merge
        # But with battery/time constraints, we need to check feasibility

        # Simpler approach: iteratively try to merge pairs of routes
        improved = True
        while improved and not time_limit_exceeded():
            improved = False
            # Compute all pairwise savings
            n_routes = len(routes)
            savings_list = []
            for i in range(n_routes):
                for j in range(i + 1, n_routes):
                    # Try all 4 combinations of endpoints
                    # routes[i] ends at routes[i][-1], routes[j] ends at routes[j][-1]
                    # We can try: i + j, i + j_reversed, i_reversed + j, i_reversed + j_reversed
                    best_combo = None
                    best_saving = -float('inf')
                    ri = routes[i]
                    rj = routes[j]
                    for rev_i in [False, True]:
                        for rev_j in [False, True]:
                            ri_t = ri[::-1] if rev_i else ri
                            rj_t = rj[::-1] if rev_j else rj
                            # Check demand
                            demand_i = sum(demands[x] for x in ri if x not in station_set)
                            demand_j = sum(demands[x] for x in rj if x not in station_set)
                            if demand_i + demand_j > capacity + 1e-9:
                                continue
                            # Compute saving: dist(0, ri[-1]) + dist(0, rj[0]) - dist(ri[-1], rj[0])
                            # But ri[-1] and rj[0] are the junction points
                            # Actually the saving is: old cost had dist(0,ri[-1])+dist(ri[-1],0) and dist(0,rj[0])+dist(rj[0],0)
                            # New cost has dist(0,ri[-1]) + dist(ri[-1],rj[0]) + ... + dist(rj[-1],0)
                            # Saving = dist(ri[-1],0) + dist(0,rj[0]) - dist(ri[-1],rj[0])
                            saving = dist[ri_t[-1]][0] + dist[0][rj_t[0]] - dist[ri_t[-1]][rj_t[0]]
                            if saving > best_saving:
                                best_saving = saving
                                best_combo = (rev_i, rev_j)

                    if best_combo is not None and best_saving > 1e-9:
                        savings_list.append((best_saving, i, j, best_combo))

            if not savings_list:
                break

            # Sort by saving descending
            savings_list.sort(key=lambda x: -x[0])

            for saving, i, j, (rev_i, rev_j) in savings_list:
                if time_limit_exceeded():
                    break
                if i >= len(routes) or j >= len(routes):
                    continue
                # Check if i and j are still valid (not already merged)
                ri = routes[i]
                rj = routes[j]
                if not ri or not rj:
                    continue

                demand_i = sum(demands[x] for x in ri if x not in station_set)
                demand_j = sum(demands[x] for x in rj if x not in station_set)
                if demand_i + demand_j > capacity + 1e-9:
                    continue

                ri_t = ri[::-1] if rev_i else ri
                rj_t = rj[::-1] if rev_j else rj

                # Try to construct the merged route with stations
                combined_customers = ri_t + rj_t
                # Filter to just customers for construct_route
                merged = construct_route(combined_customers)

                # Check if all customers from both routes are in merged
                merged_set = set(x for x in merged if x not in station_set)
                ri_set = set(x for x in ri if x not in station_set)
                rj_set = set(x for x in rj if x not in station_set)
                if merged_set != (ri_set | rj_set):
                    continue

                if is_feasible(merged):
                    # Replace routes i and j with merged
                    old_total = route_dist(ri) + route_dist(rj)
                    new_total = route_dist(merged)
                    if new_total < old_total - 1e-9:
                        # Remove the smaller index first
                        min_idx = min(i, j)
                        max_idx = max(i, j)
                        routes.pop(max_idx)
                        routes.pop(min_idx)
                        routes.append(merged)
                        improved = True

        return routes

    def two_opt(route, max_iter=50):
        """Improve route using 2-opt with station-aware feasibility."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = simulate(best_route)
        if not valid:
            return route

        n = len(best_route)
        for _ in range(max_iter):
            if time_limit_exceeded():
                break
            improved = False
            for i in range(n - 1):
                if time_limit_exceeded():
                    break
                for j in range(i + 1, n):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    valid, new_dist = simulate(new_route)
                    if valid and new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best_route

    def or_opt(route, max_iter=30):
        """OR-opt: move segments of length 1-3."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = simulate(best_route)
        if not valid:
            return route

        for _ in range(max_iter):
            if time_limit_exceeded():
                break
            improved = False
            n = len(best_route)
            for seg_len in [1, 2, 3]:
                if time_limit_exceeded():
                    break
                for i in range(n - seg_len + 1):
                    segment = best_route[i:i+seg_len]
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    for j in range(len(remaining) + 1):
                        if j == i or j == i + seg_len:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        valid, new_dist = simulate(new_route)
                        if valid and new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return best_route

    def relocate_between(routes, max_iter=20):
        """Move customers between routes."""
        for _ in range(max_iter):
            if time_limit_exceeded():
                break
            improved = False
            for i in range(len(routes)):
                if time_limit_exceeded():
                    break
                for j in range(len(routes[i])):
                    c = routes[i][j]
                    if c in station_set:
                        continue
                    new_route_i = routes[i][:j] + routes[i][j+1:]
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            new_route_k = routes[k][:pos] + [c] + routes[k][pos:]
                            valid_i, dist_i = simulate(new_route_i)
                            valid_k, dist_k = simulate(new_route_k)
                            if valid_i and valid_k:
                                old_total = route_dist(routes[i]) + route_dist(routes[k])
                                new_total = dist_i + dist_k
                                if new_total < old_total - 1e-9:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
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

    def swap_between(routes, max_iter=10):
        """Swap customers between routes."""
        for _ in range(max_iter):
            if time_limit_exceeded():
                break
            improved = False
            for i in range(len(routes)):
                if time_limit_exceeded():
                    break
                for j in range(len(routes[i])):
                    c1 = routes[i][j]
                    if c1 in station_set:
                        continue
                    for k in range(i + 1, len(routes)):
                        for pos in range(len(routes[k])):
                            c2 = routes[k][pos]
                            if c2 in station_set:
                                continue
                            new_route_i = routes[i][:]
                            new_route_i[j] = c2
                            new_route_k = routes[k][:]
                            new_route_k[pos] = c1
                            valid_i, dist_i = simulate(new_route_i)
                            valid_k, dist_k = simulate(new_route_k)
                            if valid_i and valid_k:
                                old_total = route_dist(routes[i]) + route_dist(routes[k])
                                new_total = dist_i + dist_k
                                if new_total < old_total - 1e-9:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
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

    def merge_routes(routes, max_iter=10):
        """Try merging two routes if feasible and beneficial."""
        for _ in range(max_iter):
            if time_limit_exceeded():
                break
            improved = False
            n = len(routes)
            for i in range(n):
                if time_limit_exceeded():
                    break
                for j in range(i + 1, n):
                    if j >= len(routes):
                        continue
                    for order in [0, 1]:
                        if order == 0:
                            combined_customers = routes[i] + routes[j]
                        else:
                            combined_customers = routes[j] + routes[i]
                        # Check demand
                        total_demand = sum(demands[x] for x in combined_customers if x not in station_set)
                        if total_demand > capacity + 1e-9:
                            continue
                        merged = construct_route(combined_customers)
                        merged_set = set(x for x in merged if x not in station_set)
                        expected_set = set(x for x in combined_customers if x not in station_set)
                        if merged_set != expected_set:
                            continue
                        if is_feasible(merged):
                            old_total = route_dist(routes[i]) + route_dist(routes[j])
                            new_total = route_dist(merged)
                            if new_total < old_total - 1e-9:
                                min_idx = min(i, j)
                                max_idx = max(i, j)
                                routes.pop(max_idx)
                                routes.pop(min_idx)
                                routes.append(merged)
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    # --- Generate multiple initial solutions ---
    initial_solutions = []

    # 1. Greedy by due date
    r1 = greedy_insertion("due")
    if validate_solution(r1):
        initial_solutions.append(r1)

    # 2. Greedy by angle
    r2 = greedy_insertion("angle")
    if validate_solution(r2):
        initial_solutions.append(r2)

    # 3. Greedy by nearest
    r3 = greedy_insertion("nearest")
    if validate_solution(r3):
        initial_solutions.append(r3)

    # 4. Clarke-Wright savings
    if not time_limit_exceeded():
        r4 = clarke_wright_savings()
        if validate_solution(r4):
            initial_solutions.append(r4)

    if not initial_solutions:
        # Fallback
        routes = greedy_insertion("due")
        return routes

    # Select best initial solution
    best_routes = min(initial_solutions, key=total_dist)

    # --- Local search phase ---
    # Intra-route optimization
    for r_idx in range(len(best_routes)):
        if time_limit_exceeded():
            break
        best_routes[r_idx] = two_opt(best_routes[r_idx], max_iter=30)
        if time_limit_exceeded():
            break
        best_routes[r_idx] = or_opt(best_routes[r_idx], max_iter=20)

    # Inter-route optimization
    if not time_limit_exceeded():
        best_routes = relocate_between(best_routes, max_iter=15)
    if not time_limit_exceeded():
        best_routes = swap_between(best_routes, max_iter=8)
    if not time_limit_exceeded():
        best_routes = merge_routes(best_routes, max_iter=8)

    # Second round of intra-route
    for r_idx in range(len(best_routes)):
        if time_limit_exceeded():
            break
        best_routes[r_idx] = two_opt(best_routes[r_idx], max_iter=20)
        if time_limit_exceeded():
            break
        best_routes[r_idx] = or_opt(best_routes[r_idx], max_iter=15)

    # Another inter-route pass
    if not time_limit_exceeded():
        best_routes = relocate_between(best_routes, max_iter=10)
    if not time_limit_exceeded():
        best_routes = merge_routes(best_routes, max_iter=5)

    # Final validation
    if not validate_solution(best_routes):
        # Fallback
        routes = greedy_insertion("due")
        return routes

    return best_routes

# EVOLVE-BLOCK-END
