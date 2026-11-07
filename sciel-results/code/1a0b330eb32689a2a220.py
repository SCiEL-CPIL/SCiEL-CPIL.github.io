# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from math import sqrt

def plan(instance):
    start_time = time.time()
    DEADLINE = 118.0

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)

    # Precompute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    customers = list(range(1, n_customers + 1))
    station_list = list(stations)

    def is_feasible(route):
        """Check if a route is feasible. Returns (valid, total_distance)."""
        if not route:
            return True, 0.0

        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        total_demand = 0.0

        for node in route:
            d = dist[curr_node][node]
            total_dist += d

            if curr_battery < d - 1e-9:
                return False, 0.0

            curr_battery -= d
            curr_time += d / speed

            if node in stations:
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
            else:
                total_demand += demands[node]
                if total_demand > capacity + 1e-9:
                    return False, 0.0

                service_start = max(curr_time, ready[node])
                if service_start > due[node] + 1e-9:
                    return False, 0.0

                curr_time = service_start + service[node]

            curr_node = node

        d = dist[curr_node][0]
        total_dist += d

        if curr_battery < d - 1e-9:
            return False, 0.0

        curr_time += d / speed

        if curr_time > horizon + 1e-9:
            return False, 0.0

        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        curr = 0
        total = 0.0
        for node in route:
            total += dist[curr][node]
            curr = node
        total += dist[curr][0]
        return total

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def validate_solution(routes):
        """Check all constraints."""
        visited = set()
        for route in routes:
            valid, _ = is_feasible(route)
            if not valid:
                return False
            for node in route:
                if node not in stations:
                    if node in visited:
                        return False
                    visited.add(node)
        return visited == set(customers)

    def greedy_insertion(sort_key="due"):
        """Greedy insertion heuristic."""
        routes = []
        if sort_key == "due":
            sorted_customers = sorted(customers, key=lambda v: due[v])
        elif sort_key == "angle":
            sorted_customers = sorted(customers, key=lambda v: np.arctan2(coords[v][1] - coords[0][1], coords[v][0] - coords[0][0]))
        elif sort_key == "dist":
            sorted_customers = sorted(customers, key=lambda v: dist[0][v])
        else:
            sorted_customers = customers[:]

        for c in sorted_customers:
            if time.time() - start_time > DEADLINE:
                break

            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')

            for r_idx, route in enumerate(routes):
                current_demand = sum(demands[node] for node in route if node not in stations)
                if current_demand + demands[c] > capacity + 1e-9:
                    continue

                for p in range(len(route) + 1):
                    new_route = route[:p] + [c] + route[p:]
                    valid, actual_dist = is_feasible(new_route)
                    if valid and actual_dist < best_cost:
                        best_cost = actual_dist
                        best_route_idx = r_idx
                        best_pos = p

            if best_route_idx != -1:
                routes[best_route_idx].insert(best_pos, c)
            else:
                routes.append([c])

        return routes

    def nearest_neighbor():
        """Nearest neighbor construction."""
        routes = []
        unvisited = set(customers)

        while unvisited:
            if time.time() - start_time > DEADLINE:
                break

            # Start from depot, find nearest feasible customer
            curr_node = 0
            curr_time = 0.0
            curr_battery = battery_max
            total_demand = 0.0
            route = []

            while unvisited:
                best_c = None
                best_dist = float('inf')
                for c in unvisited:
                    if total_demand + demands[c] > capacity + 1e-9:
                        continue
                    d = dist[curr_node][c]
                    if curr_battery >= d - 1e-9:
                        new_time = curr_time + d / speed
                        service_start = max(new_time, ready[c])
                        if service_start <= due[c] + 1e-9:
                            if d < best_dist:
                                best_dist = d
                                best_c = c
                    else:
                        # Try via station
                        for s in station_list:
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
                                if total_d < best_dist:
                                    best_dist = total_d
                                    best_c = c

                if best_c is None:
                    break

                # Go to best_c
                d_direct = dist[curr_node][best_c]
                if curr_battery >= d_direct - 1e-9:
                    new_time = curr_time + d_direct / speed
                    service_start = max(new_time, ready[best_c])
                    if service_start <= due[best_c] + 1e-9:
                        route.append(best_c)
                        curr_battery -= d_direct
                        curr_time = service_start + service[best_c]
                        total_demand += demands[best_c]
                        curr_node = best_c
                        unvisited.remove(best_c)
                        continue

                # Need station
                best_s = None
                best_total_d = float('inf')
                for s in station_list:
                    d1 = dist[curr_node][s]
                    if curr_battery < d1 - 1e-9:
                        continue
                    charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                    new_time = curr_time + d1 / speed + charge_time
                    d2 = dist[s][best_c]
                    new_time2 = new_time + d2 / speed
                    service_start = max(new_time2, ready[best_c])
                    if service_start <= due[best_c] + 1e-9:
                        total_d = d1 + d2
                        if total_d < best_total_d:
                            best_total_d = total_d
                            best_s = s

                if best_s is None:
                    break

                s = best_s
                d1 = dist[curr_node][s]
                curr_battery -= d1
                curr_time += d1 / speed
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
                route.append(s)
                curr_node = s

                d2 = dist[curr_node][best_c]
                curr_battery -= d2
                curr_time += d2 / speed
                service_start = max(curr_time, ready[best_c])
                curr_time = service_start + service[best_c]
                total_demand += demands[best_c]
                curr_node = best_c
                route.append(best_c)
                unvisited.remove(best_c)

            if route:
                routes.append(route)

        return routes

    def cluster_based():
        """Cluster customers by angle and build routes per cluster."""
        angles = [np.arctan2(coords[c][1] - coords[0][1], coords[c][0] - coords[0][0]) for c in customers]
        angle_order = [customers[i] for i in np.argsort(angles)]

        routes = []
        cluster = []
        cluster_demand = 0.0

        for c in angle_order:
            if time.time() - start_time > DEADLINE:
                break
            if cluster_demand + demands[c] > capacity:
                if cluster:
                    # Build route for this cluster
                    route = []
                    curr_node = 0
                    curr_time = 0.0
                    curr_battery = battery_max
                    total_demand = 0.0
                    # Sort cluster by nearest neighbor
                    remaining = cluster[:]
                    while remaining:
                        best_c = None
                        best_d = float('inf')
                        for c2 in remaining:
                            d = dist[curr_node][c2]
                            if d < best_d:
                                best_d = d
                                best_c = c2
                        if best_c is None:
                            break
                        if curr_battery >= best_d - 1e-9:
                            new_time = curr_time + best_d / speed
                            service_start = max(new_time, ready[best_c])
                            if service_start <= due[best_c] + 1e-9:
                                route.append(best_c)
                                curr_battery -= best_d
                                curr_time = service_start + service[best_c]
                                total_demand += demands[best_c]
                                curr_node = best_c
                                remaining.remove(best_c)
                                continue
                        # Try station
                        best_s = None
                        best_sd = float('inf')
                        for s in station_list:
                            d1 = dist[curr_node][s]
                            if curr_battery < d1 - 1e-9:
                                continue
                            d2 = dist[s][best_c]
                            total_sd = d1 + d2
                            if total_sd < best_sd:
                                best_sd = total_sd
                                best_s = s
                        if best_s is None:
                            break
                        s = best_s
                        d1 = dist[curr_node][s]
                        curr_battery -= d1
                        curr_time += d1 / speed
                        charge_time = (battery_max - curr_battery) * recharge_rate
                        curr_time += charge_time
                        curr_battery = battery_max
                        route.append(s)
                        curr_node = s
                        d2 = dist[curr_node][best_c]
                        curr_battery -= d2
                        curr_time += d2 / speed
                        service_start = max(curr_time, ready[best_c])
                        curr_time = service_start + service[best_c]
                        total_demand += demands[best_c]
                        curr_node = best_c
                        route.append(best_c)
                        remaining.remove(best_c)
                    if route:
                        routes.append(route)
                    cluster = []
                    cluster_demand = 0.0
            cluster.append(c)
            cluster_demand += demands[c]

        if cluster:
            route = []
            curr_node = 0
            curr_time = 0.0
            curr_battery = battery_max
            total_demand = 0.0
            remaining = cluster[:]
            while remaining:
                best_c = None
                best_d = float('inf')
                for c2 in remaining:
                    d = dist[curr_node][c2]
                    if d < best_d:
                        best_d = d
                        best_c = c2
                if best_c is None:
                    break
                if curr_battery >= best_d - 1e-9:
                    new_time = curr_time + best_d / speed
                    service_start = max(new_time, ready[best_c])
                    if service_start <= due[best_c] + 1e-9:
                        route.append(best_c)
                        curr_battery -= best_d
                        curr_time = service_start + service[best_c]
                        total_demand += demands[best_c]
                        curr_node = best_c
                        remaining.remove(best_c)
                        continue
                best_s = None
                best_sd = float('inf')
                for s in station_list:
                    d1 = dist[curr_node][s]
                    if curr_battery < d1 - 1e-9:
                        continue
                    d2 = dist[s][best_c]
                    total_sd = d1 + d2
                    if total_sd < best_sd:
                        best_sd = total_sd
                        best_s = s
                if best_s is None:
                    break
                s = best_s
                d1 = dist[curr_node][s]
                curr_battery -= d1
                curr_time += d1 / speed
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
                route.append(s)
                curr_node = s
                d2 = dist[curr_node][best_c]
                curr_battery -= d2
                curr_time += d2 / speed
                service_start = max(curr_time, ready[best_c])
                curr_time = service_start + service[best_c]
                total_demand += demands[best_c]
                curr_node = best_c
                route.append(best_c)
                remaining.remove(best_c)
            if route:
                routes.append(route)

        # Handle any unvisited
        visited = set()
        for r in routes:
            for node in r:
                if node not in stations:
                    visited.add(node)
        for c in customers:
            if c not in visited:
                best_route_idx = -1
                best_pos = -1
                best_cost = float('inf')
                for r_idx, route in enumerate(routes):
                    current_demand = sum(demands[node] for node in route if node not in stations)
                    if current_demand + demands[c] > capacity + 1e-9:
                        continue
                    for p in range(len(route) + 1):
                        new_route = route[:p] + [c] + route[p:]
                        valid, actual_dist = is_feasible(new_route)
                        if valid and actual_dist < best_cost:
                            best_cost = actual_dist
                            best_route_idx = r_idx
                            best_pos = p
                if best_route_idx != -1:
                    routes[best_route_idx].insert(best_pos, c)
                else:
                    routes.append([c])

        return routes

    def two_opt(route):
        """Improve route using 2-opt."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return route

        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for i in range(n - 1):
                if time.time() - start_time > DEADLINE:
                    return best_route
                for j in range(i + 1, n):
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    valid, new_dist = is_feasible(new_route)
                    if valid and new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best_route

    def or_opt(route):
        """Improve route using OR-opt (move 1-2 nodes)."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return route

        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for seg_len in [1, 2]:
                for i in range(n - seg_len + 1):
                    if time.time() - start_time > DEADLINE:
                        return best_route
                    segment = best_route[i:i+seg_len]
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    for j in range(len(remaining) + 1):
                        new_route = remaining[:j] + segment + remaining[j:]
                        if new_route == best_route:
                            continue
                        valid, new_dist = is_feasible(new_route)
                        if valid and new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    def relocate_improve(routes):
        """Try moving customers between routes."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                break
            for i in range(len(routes)):
                for j in range(len(routes[i])):
                    c = routes[i][j]
                    new_route_i = routes[i][:j] + routes[i][j+1:]
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            new_route_k = routes[k][:pos] + [c] + routes[k][pos:]
                            valid_i, dist_i = is_feasible(new_route_i)
                            valid_k, dist_k = is_feasible(new_route_k)
                            if valid_i and valid_k:
                                old_total = route_distance(routes[i]) + route_distance(routes[k])
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
        return routes

    def merge_routes(routes):
        """Try merging two routes if feasible."""
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > DEADLINE:
                break
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    for order in [0, 1]:
                        if order == 0:
                            combined = routes[i] + routes[j]
                        else:
                            combined = routes[j] + routes[i]
                        valid, total_dist = is_feasible(combined)
                        if valid:
                            old_total = route_distance(routes[i]) + route_distance(routes[j])
                            if total_dist < old_total - 1e-9:
                                new_routes = routes[:i] + [combined] + routes[i+1:j] + routes[j+1:]
                                routes = new_routes
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Generate multiple initial solutions
    initial_solutions = []

    r1 = greedy_insertion("due")
    if validate_solution(r1):
        initial_solutions.append(r1)

    r2 = greedy_insertion("angle")
    if validate_solution(r2):
        initial_solutions.append(r2)

    r3 = greedy_insertion("dist")
    if validate_solution(r3):
        initial_solutions.append(r3)

    r4 = nearest_neighbor()
    if validate_solution(r4):
        initial_solutions.append(r4)

    r5 = cluster_based()
    if validate_solution(r5):
        initial_solutions.append(r5)

    if not initial_solutions:
        routes = greedy_insertion("due")
        return routes

    # Select best initial solution
    best_routes = min(initial_solutions, key=total_distance)

    # Local search
    for r_idx in range(len(best_routes)):
        if time.time() - start_time > DEADLINE:
            break
        best_routes[r_idx] = two_opt(best_routes[r_idx])
        if time.time() - start_time > DEADLINE:
            break
        best_routes[r_idx] = or_opt(best_routes[r_idx])

    # Inter-route improvements
    if time.time() - start_time < DEADLINE:
        best_routes = relocate_improve(best_routes)

    if time.time() - start_time < DEADLINE:
        best_routes = merge_routes(best_routes)

    # Final 2-opt pass
    for r_idx in range(len(best_routes)):
        if time.time() - start_time > DEADLINE:
            break
        best_routes[r_idx] = two_opt(best_routes[r_idx])

    # Final validation
    if not validate_solution(best_routes):
        routes = greedy_insertion("due")
        return routes

    return best_routes

# EVOLVE-BLOCK-END
