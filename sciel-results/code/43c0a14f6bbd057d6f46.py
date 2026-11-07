"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
import heapq
from itertools import combinations


def _build_dist_matrix(coords):
    """Build Euclidean distance matrix using NumPy."""
    n = len(coords)
    c = np.array(coords, dtype=np.float64)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """
    Simulate a single route. Returns (total_distance, is_valid).
    route: list of node ids (customers and/or stations), depot implicit at start/end.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_dist = 0.0
    t = 0.0
    batt = battery_max
    load = 0
    prev = 0  # depot

    for node in route:
        d = dist[prev][node]
        total_dist += d
        batt -= d
        t += d / speed

        if batt < -1e-9:
            return total_dist, False

        load += demands[node]
        if load > capacity:
            return total_dist, False

        if node in stations_set:
            # Recharge
            if batt < 0:
                batt = 0.0
            energy = battery_max - batt
            t += energy * recharge_rate
            batt = battery_max
        else:
            # Customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False
            t += service[node]

        prev = node

    # Return to depot
    d = dist[prev][0]
    total_dist += d
    batt -= d
    t += d / speed

    if batt < -1e-9:
        return total_dist, False
    if t > horizon + 1e-9:
        return total_dist, False

    return total_dist, True


def _validate_solution(routes, dist, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity):
    """Validate that all customers are served exactly once and all routes are feasible."""
    served = set()
    total_dist = 0.0
    for r in routes:
        d, valid = _simulate_route(r, dist, instance, n_customers, stations_set,
                                   battery_max, speed, recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_dist += d
        for node in r:
            if node in stations_set:
                continue
            if node in served:
                return False, 0.0
            served.add(node)

    all_customers = set(range(1, n_customers + 1))
    if served != all_customers:
        return False, 0.0

    return True, total_dist


def _compute_total_distance(routes, dist):
    """Compute total distance of all routes."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        prev = 0
        for node in r:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
    return total


def _route_feasible(route, dist, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """Quick check if a route is feasible."""
    _, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                               battery_max, speed, recharge_rate, horizon, capacity)
    return valid


def _nearest_neighbor_construct(customers, dist, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity):
    """
    Construct routes using nearest neighbor heuristic with battery/time window checks.
    """
    routes = []
    remaining = set(customers)

    while remaining:
        # Start with the customer with the earliest due time
        start_cust = min(remaining, key=lambda c: instance["due"][c])
        route = [start_cust]
        remaining.discard(start_cust)

        current_node = start_cust
        current_battery = battery_max - dist[0][start_cust]
        current_time = dist[0][start_cust] / speed
        if current_time < instance["ready"][start_cust]:
            current_time = instance["ready"][start_cust]
        current_time += instance["service"][start_cust]
        current_load = instance["demands"][start_cust]

        while remaining:
            # Find the nearest feasible customer
            best_cust = None
            best_dist = float('inf')

            for cust in remaining:
                d = dist[current_node][cust]
                if current_battery - d < -1e-9:
                    continue
                if current_load + instance["demands"][cust] > capacity:
                    continue

                t_arrive = current_time + d / speed
                if t_arrive > instance["due"][cust] + 1e-9:
                    continue

                # Check if we can return to depot after visiting cust
                d_back = dist[cust][0]
                if current_battery - d - d_back < -1e-9:
                    continue

                if d < best_dist:
                    best_dist = d
                    best_cust = cust

            if best_cust is None:
                break

            route.append(best_cust)
            remaining.discard(best_cust)

            d = dist[current_node][best_cust]
            current_battery -= d
            current_time += d / speed
            if current_time < instance["ready"][best_cust]:
                current_time = instance["ready"][best_cust]
            current_time += instance["service"][best_cust]
            current_load += instance["demands"][best_cust]
            current_node = best_cust

        # Check if route is feasible
        d_total, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity)
        if not valid:
            # Try to split the route
            for i in range(len(route) - 1, 0, -1):
                sub_route = route[:i]
                d_sub, valid_sub = _simulate_route(sub_route, dist, instance, n_customers,
                                                   stations_set, battery_max, speed,
                                                   recharge_rate, horizon, capacity)
                if valid_sub:
                    routes.append(sub_route)
                    # Add remaining customers as separate routes
                    for cust in route[i:]:
                        routes.append([cust])
                    break
            else:
                # All individual customers
                for cust in route:
                    routes.append([cust])
        else:
            routes.append(route)

    return routes


def _savings_construct(customers, dist, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity):
    """
    Construct routes using Clarke-Wright savings heuristic.
    """
    routes = [[c] for c in customers]
    route_of = {c: i for i, r in enumerate(routes) for c in r}

    # Compute savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            s = dist[0][i] + dist[0][j] - dist[i][j]
            savings.append((-s, i, j))
    savings.sort()

    for neg_s, i, j in savings:
        if i not in route_of or j not in route_of:
            continue
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Check if i is at the end of route_i or j is at the end of route_j
        # We can merge if i is last in route_i and j is first in route_j,
        # or j is last in route_j and i is first in route_i

        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_j[-1] == j and route_i[-1] == i:
            merged = route_j + route_i[::-1]

        if merged is None:
            continue

        # Check feasibility
        d_total, valid = _simulate_route(merged, dist, instance, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity)
        if not valid:
            continue

        # Merge
        routes[ri] = merged
        for c in merged:
            route_of[c] = ri
        routes[rj] = []

    return [r for r in routes if r]


def _insert_stations(route, dist, instance, n_customers, stations_set,
                     battery_max, speed, recharge_rate, horizon, capacity):
    """
    Try to insert stations into a route to make it feasible.
    Returns the modified route or None if infeasible.
    """
    # First check if the route is already feasible
    d_total, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity)
    if valid:
        return route

    # Try inserting stations at various positions
    best_route = None
    best_dist = float('inf')

    # Try inserting one station
    for st in stations_set:
        for pos in range(len(route) + 1):
            new_route = route[:pos] + [st] + route[pos:]
            d_new, valid_new = _simulate_route(new_route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
            if valid_new and d_new < best_dist:
                best_dist = d_new
                best_route = new_route

    if best_route is not None:
        return best_route

    # Try inserting two stations
    for st1 in stations_set:
        for st2 in stations_set:
            if st1 == st2:
                continue
            for pos1 in range(len(route) + 1):
                for pos2 in range(pos1 + 1, len(route) + 2):
                    new_route = route[:pos1] + [st1] + route[pos1:pos2] + [st2] + route[pos2:]
                    d_new, valid_new = _simulate_route(new_route, dist, instance, n_customers,
                                                       stations_set, battery_max, speed,
                                                       recharge_rate, horizon, capacity)
                    if valid_new and d_new < best_dist:
                        best_dist = d_new
                        best_route = new_route

    return best_route


def _two_opt(route, dist, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity):
    """Apply 2-opt improvement to a single route."""
    n = len(route)
    if n < 3:
        return route

    d_old, valid_old = _simulate_route(route, dist, instance, n_customers,
                                       stations_set, battery_max, speed,
                                       recharge_rate, horizon, capacity)
    if not valid_old:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d_new, valid = _simulate_route(new_route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
                if valid and d_new < d_old - 1e-9:
                    route = new_route
                    d_old = d_new
                    improved = True
    return route


def _or_opt(route, dist, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity):
    """Apply OR-opt improvement to a single route (relocate 1-2 consecutive nodes)."""
    n = len(route)
    if n < 3:
        return route

    d_old, valid_old = _simulate_route(route, dist, instance, n_customers,
                                       stations_set, battery_max, speed,
                                       recharge_rate, horizon, capacity)
    if not valid_old:
        return route

    improved = True
    while improved:
        improved = False
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                segment = route[i:i + seg_len]
                rest = route[:i] + route[i + seg_len:]
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + segment + rest[j:]
                    if new_route == route:
                        continue
                    d_new, valid = _simulate_route(new_route, dist, instance, n_customers,
                                                   stations_set, battery_max, speed,
                                                   recharge_rate, horizon, capacity)
                    if valid and d_new < d_old - 1e-9:
                        route = new_route
                        d_old = d_new
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route


def _local_search(routes, dist, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity,
                  time_limit=30.0):
    """Apply 2-opt, OR-opt, and relocate moves."""
    start = time.time()
    improved = True

    while improved and (time.time() - start) < time_limit:
        improved = False

        # 2-opt within each route
        for ri in range(len(routes)):
            if (time.time() - start) > time_limit:
                return routes
            routes[ri] = _two_opt(
                routes[ri], dist, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity
            )

        # OR-opt within each route
        for ri in range(len(routes)):
            if (time.time() - start) > time_limit:
                return routes
            routes[ri] = _or_opt(
                routes[ri], dist, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity
            )

        # Relocate moves
        for ri in range(len(routes)):
            for pos in range(len(routes[ri])):
                if (time.time() - start) > time_limit:
                    return routes
                cust = routes[ri][pos]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos2 in range(len(routes[rj]) + 1):
                        new_ri = routes[ri][:pos] + routes[ri][pos + 1:]
                        new_rj = routes[rj][:pos2] + [cust] + routes[rj][pos2:]

                        # Check feasibility
                        valid_ri = True
                        valid_rj = True
                        d_ri = 0.0
                        d_rj = 0.0

                        if new_ri:
                            d_ri, valid_ri = _simulate_route(new_ri, dist, instance,
                                                             n_customers, stations_set,
                                                             battery_max, speed,
                                                             recharge_rate, horizon, capacity)
                        if new_rj:
                            d_rj, valid_rj = _simulate_route(new_rj, dist, instance,
                                                             n_customers, stations_set,
                                                             battery_max, speed,
                                                             recharge_rate, horizon, capacity)

                        if valid_ri and valid_rj:
                            old_total = _compute_total_distance([routes[ri], routes[rj]], dist)
                            new_total = d_ri + d_rj
                            if new_total < old_total - 1e-9:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    return routes


def _simulated_annealing(routes, dist, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity,
                         time_limit=20.0):
    """Apply simulated annealing to improve routes."""
    start = time.time()
    current_routes = [list(r) for r in routes]
    current_dist = _compute_total_distance(current_routes, dist)
    best_routes = [list(r) for r in current_routes]
    best_dist = current_dist

    T = 50.0
    cooling = 0.995

    while (time.time() - start) < time_limit and T > 0.1:
        # Generate a random perturbation
        new_routes = [list(r) for r in current_routes]
        move_type = random.random()

        if move_type < 0.4:
            # 2-opt within a route
            ri = random.randint(0, len(new_routes) - 1)
            route = new_routes[ri]
            if len(route) >= 3:
                i, j = sorted(random.sample(range(len(route)), 2))
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d, valid = _simulate_route(new_route, dist, instance, n_customers,
                                           stations_set, battery_max, speed,
                                           recharge_rate, horizon, capacity)
                if valid:
                    new_routes[ri] = new_route
        elif move_type < 0.7:
            # OR-opt within a route
            ri = random.randint(0, len(new_routes) - 1)
            route = new_routes[ri]
            if len(route) >= 3:
                seg_len = random.choice([1, 2])
                if len(route) >= seg_len:
                    i = random.randint(0, len(route) - seg_len)
                    segment = route[i:i + seg_len]
                    rest = route[:i] + route[i + seg_len:]
                    j = random.randint(0, len(rest))
                    new_route = rest[:j] + segment + rest[j:]
                    d, valid = _simulate_route(new_route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
                    if valid:
                        new_routes[ri] = new_route
        else:
            # Relocate a customer
            ri = random.randint(0, len(new_routes) - 1)
            if len(new_routes[ri]) > 0:
                pos = random.randint(0, len(new_routes[ri]) - 1)
                cust = new_routes[ri][pos]
                rj = random.randint(0, len(new_routes) - 1)
                pos2 = random.randint(0, len(new_routes[rj]))
                new_ri = new_routes[ri][:pos] + new_routes[ri][pos + 1:]
                new_rj = new_routes[rj][:pos2] + [cust] + new_routes[rj][pos2:]

                valid_ri = True
                valid_rj = True
                d_ri = 0.0
                d_rj = 0.0
                if new_ri:
                    d_ri, valid_ri = _simulate_route(new_ri, dist, instance, n_customers,
                                                     stations_set, battery_max, speed,
                                                     recharge_rate, horizon, capacity)
                if new_rj:
                    d_rj, valid_rj = _simulate_route(new_rj, dist, instance, n_customers,
                                                     stations_set, battery_max, speed,
                                                     recharge_rate, horizon, capacity)

                if valid_ri and valid_rj:
                    new_routes[ri] = new_ri
                    new_routes[rj] = new_rj

        new_dist = _compute_total_distance(new_routes, dist)
        delta = new_dist - current_dist

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_routes = [list(r) for r in current_routes]
                best_dist = current_dist

        T *= cooling

    return best_routes


def _cluster_construct(customers, dist, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity,
                       n_clusters=4):
    """
    Cluster customers into groups and construct routes for each cluster.
    """
    # Simple k-means-like clustering
    # Initialize cluster centers as evenly spaced customers
    indices = list(range(n_customers))
    centers = [indices[i * len(indices) // n_clusters] for i in range(n_clusters)]

    # Assign customers to nearest center
    clusters = [[] for _ in range(n_clusters)]
    for c in customers:
        min_d = float('inf')
        min_cluster = 0
        for ci, center in enumerate(centers):
            d = dist[c][center]
            if d < min_d:
                min_d = d
                min_cluster = ci
        clusters[min_cluster].append(c)

    # Build routes for each cluster
    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        # Use nearest neighbor within cluster
        route = [cluster[0]]
        remaining = set(cluster[1:])

        current_node = cluster[0]
        current_battery = battery_max - dist[0][cluster[0]]
        current_time = dist[0][cluster[0]] / speed
        if current_time < instance["ready"][cluster[0]]:
            current_time = instance["ready"][cluster[0]]
        current_time += instance["service"][cluster[0]]
        current_load = instance["demands"][cluster[0]]

        while remaining:
            best_cust = None
            best_dist = float('inf')

            for cust in remaining:
                d = dist[current_node][cust]
                if current_battery - d < -1e-9:
                    continue
                if current_load + instance["demands"][cust] > capacity:
                    continue

                t_arrive = current_time + d / speed
                if t_arrive > instance["due"][cust] + 1e-9:
                    continue

                d_back = dist[cust][0]
                if current_battery - d - d_back < -1e-9:
                    continue

                if d < best_dist:
                    best_dist = d
                    best_cust = cust

            if best_cust is None:
                break

            route.append(best_cust)
            remaining.discard(best_cust)

            d = dist[current_node][best_cust]
            current_battery -= d
            current_time += d / speed
            if current_time < instance["ready"][best_cust]:
                current_time = instance["ready"][best_cust]
            current_time += instance["service"][best_cust]
            current_load += instance["demands"][best_cust]
            current_node = best_cust

        d_total, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                                         battery_max, speed, recharge_rate, horizon, capacity)
        if not valid:
            # Try inserting stations
            modified = _insert_stations(route, dist, instance, n_customers, stations_set,
                                        battery_max, speed, recharge_rate, horizon, capacity)
            if modified is not None:
                route = modified
            else:
                # Fall back to individual customers
                for c in route:
                    routes.append([c])
                continue

        routes.append(route)

    return routes


def plan(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))

    # Build distance matrix
    dist = _build_dist_matrix(coords)

    # Deadline guard
    deadline = time.time() + 140.0

    # Phase 1: Try multiple construction heuristics and pick the best
    best_routes = None
    best_dist = float('inf')

    # Try nearest neighbor
    routes_nn = _nearest_neighbor_construct(
        customers, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )
    valid, d_nn = _validate_solution(
        routes_nn, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )
    if valid and d_nn < best_dist:
        best_dist = d_nn
        best_routes = routes_nn

    # Try savings
    if time.time() < deadline - 30.0:
        routes_sv = _savings_construct(
            customers, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity
        )
        valid, d_sv = _validate_solution(
            routes_sv, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity
        )
        if valid and d_sv < best_dist:
            best_dist = d_sv
            best_routes = routes_sv

    # Try cluster
    if time.time() < deadline - 30.0:
        routes_cl = _cluster_construct(
            customers, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            n_clusters=4
        )
        valid, d_cl = _validate_solution(
            routes_cl, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity
        )
        if valid and d_cl < best_dist:
            best_dist = d_cl
            best_routes = routes_cl

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[v] for v in customers]

    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        best_routes = _local_search(
            best_routes, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            time_limit=max(1.0, deadline - time.time() - 10.0)
        )

    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        best_routes = _simulated_annealing(
            best_routes, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            time_limit=max(1.0, deadline - time.time() - 5.0)
        )

    # Phase 4: Final polish with tighter local search
    if time.time() < deadline - 2.0:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 1.0:
            best_routes = _local_search(
                best_routes, dist, instance, n_customers, stations,
                battery_cap, speed, recharge_rate, horizon, capacity,
                time_limit=remaining_time
            )

    # Final validation
    valid, total_dist = _validate_solution(
        best_routes, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )
    if not valid:
        best_routes = [[v] for v in customers]

    return best_routes
# EVOLVE-BLOCK-END
