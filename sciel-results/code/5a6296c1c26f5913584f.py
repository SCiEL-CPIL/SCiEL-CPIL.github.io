"""EV routing with time windows and battery constraints - savings-based construction with station insertion."""
import numpy as np
import math
import time
import random

DEADLINE_SECONDS = 142.0


def _build_dist_matrix(coords):
    coords = np.array(coords, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations_set:
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time_at += energy_needed * recharge_rate
            battery = battery_max

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d
    distance += d

    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0

    return True, distance, time_at, battery


def _validate_solution(routes, instance):
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    dist_matrix = _build_dist_matrix(coords)
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    served = set()
    total_distance = 0.0

    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                             stations_set, battery_max, speed,
                                             recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_distance += dist
        for node in route:
            if 1 <= node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed, recharge_rate,
                                      horizon, capacity)
    return valid


def _try_insert_station(route, pos, station, dist_matrix, instance, n_customers,
                        stations_set, battery_max, speed, recharge_rate, horizon, capacity):
    """Try inserting a station at position pos in route. Returns new route if feasible."""
    new_route = route[:pos] + [station] + route[pos:]
    if _route_feasible(new_route, dist_matrix, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity):
        return new_route
    return None


def _savings_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """
    Clarke-Wright savings heuristic with automatic station insertion.
    """
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    # Initialize: each customer in its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_loads = [demands[i] for i in range(1, n_customers + 1)]

    # Compute savings: savings(i, j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Map: which route contains each customer
    cust_route = {}
    for r_idx, route in enumerate(routes):
        for c in route:
            cust_route[c] = r_idx

    for s, i, j in savings:
        if s <= 0:
            break
        if i not in cust_route or j not in cust_route:
            continue
        r1 = cust_route[i]
        r2 = cust_route[j]
        if r1 == r2:
            continue

        # Check capacity
        if route_loads[r1] + route_loads[r2] > capacity:
            continue

        # Check if i is at an end of route r1 and j is at an end of route r2
        # (or vice versa)
        route1 = routes[r1]
        route2 = routes[r2]

        merged = None
        # Try: i at end of route1, j at start of route2
        if route1 and route2:
            if route1[-1] == i and route2[0] == j:
                candidate = route1 + route2
                if _route_feasible(candidate, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
                    merged = candidate
            elif route1[-1] == j and route2[0] == i:
                candidate = route1 + route2[::-1]
                if _route_feasible(candidate, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
                    merged = candidate
            elif route1[0] == i and route2[-1] == j:
                candidate = route2 + route1
                if _route_feasible(candidate, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
                    merged = candidate
            elif route1[0] == j and route2[-1] == i:
                candidate = route2[::-1] + route1
                if _route_feasible(candidate, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
                    merged = candidate

        if merged is None:
            # Try merging with various orientations, and insert stations if needed
            for r1_dir, r2_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                if r1_dir == 1:
                    part1 = route1
                    end1 = route1[-1]
                else:
                    part1 = route1[::-1]
                    end1 = route1[0]

                if r2_dir == 1:
                    part2 = route2
                    start2 = route2[0]
                else:
                    part2 = route2[::-1]
                    start2 = route2[-1]

                # Check that the joining ends are i and j (in some order)
                if not ((end1 == i and start2 == j) or (end1 == j and start2 == i)):
                    continue

                # First try without station
                candidate = part1 + part2
                if _route_feasible(candidate, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity):
                    merged = candidate
                    break

                # Try inserting a station at the junction
                for st in instance["stations"]:
                    new_candidate = part1 + [st] + part2
                    if _route_feasible(new_candidate, dist_matrix, instance, n_customers,
                                       stations_set, battery_max, speed, recharge_rate,
                                       horizon, capacity):
                        merged = new_candidate
                        break
                if merged:
                    break

                # Try inserting station at other positions in the merged route
                merged_len = len(part1) + len(part2)
                for st in instance["stations"]:
                    for pos in range(1, merged_len):
                        if pos == len(part1):
                            continue  # already tried junction
                        new_candidate = part1[:pos] + [st] + part1[pos:] + part2
                        if _route_feasible(new_candidate, dist_matrix, instance, n_customers,
                                           stations_set, battery_max, speed, recharge_rate,
                                           horizon, capacity):
                            merged = new_candidate
                            break
                    if merged:
                        break
                if merged:
                    break

        if merged is not None:
            # Update routes
            new_route = merged
            routes[r1] = new_route
            routes[r2] = []
            route_loads[r1] += route_loads[r2]
            for c in new_route:
                cust_route[c] = r1

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _nearest_neighbor_construct(instance, dist_matrix, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """Nearest neighbor construction with station insertion."""
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    remaining = list(range(1, n_customers + 1))
    if random.random() < 0.5:
        remaining.sort(key=lambda x: due[x])
    else:
        random.shuffle(remaining)

    routes = []
    current_route = []
    current_load = 0
    current_node = 0
    battery = battery_max
    time_at = 0.0

    while remaining:
        best_next = None
        best_dist = float('inf')
        best_battery = 0
        best_time = 0

        for cust in remaining:
            d = dist_matrix[current_node, cust]
            if battery - d < -1e-9:
                # Try going to a station first
                for st in instance["stations"]:
                    d_st = dist_matrix[current_node, st]
                    if battery - d_st < -1e-9:
                        continue
                    bat_after_st = battery - d_st
                    t_after_st = time_at + d_st / speed
                    energy_needed = battery_max - bat_after_st
                    if energy_needed > 0:
                        t_after_st += energy_needed * recharge_rate
                    bat_after_st = battery_max
                    d_st_cust = dist_matrix[st, cust]
                    if bat_after_st - d_st_cust < -1e-9:
                        continue
                    t_after_cust = t_after_st + d_st_cust / speed
                    if t_after_cust < ready[cust]:
                        t_after_cust = ready[cust]
                    if t_after_cust > due[cust] + 1e-9:
                        continue
                    t_after_cust += instance["service"][cust]
                    total_d = d_st + d_st_cust
                    if total_d < best_dist:
                        best_dist = total_d
                        best_next = ('station_then_cust', st, cust)
                        best_battery = bat_after_st - d_st_cust
                        best_time = t_after_cust
                continue

            new_battery = battery - d
            new_time = time_at + d / speed
            if new_time < ready[cust]:
                new_time = ready[cust]
            if new_time > due[cust] + 1e-9:
                continue
            new_time += instance["service"][cust]

            if d < best_dist:
                best_dist = d
                best_next = ('cust', cust)
                best_battery = new_battery
                best_time = new_time

        if best_next is None:
            # Can't reach any remaining customer; close this route and start new
            if current_route:
                # Check if we can return to depot
                d_ret = dist_matrix[current_node, 0]
                if battery - d_ret >= -1e-9:
                    routes.append(current_route)
                else:
                    # Try adding station
                    for st in instance["stations"]:
                        d_st = dist_matrix[current_node, st]
                        if battery - d_st < -1e-9:
                            continue
                        bat_st = battery - d_st
                        t_st = time_at + d_st / speed
                        energy_needed = battery_max - bat_st
                        if energy_needed > 0:
                            t_st += energy_needed * recharge_rate
                        bat_st = battery_max
                        d_ret2 = dist_matrix[st, 0]
                        if bat_st - d_ret2 >= -1e-9:
                            routes.append(current_route + [st])
                            break
                    else:
                        # Just append what we have (may be infeasible, will be fixed later)
                        routes.append(current_route)
                current_route = []
                current_load = 0
                current_node = 0
                battery = battery_max
                time_at = 0.0
                continue

        if isinstance(best_next, tuple):
            if best_next[0] == 'station_then_cust':
                _, st, cust = best_next
                current_route.append(st)
                current_route.append(cust)
                remaining.remove(cust)
            else:
                cust = best_next[1]
                current_route.append(cust)
                remaining.remove(cust)
        else:
            current_route.append(best_next)
            remaining.remove(best_next)

        current_load += demands[best_next[1] if isinstance(best_next, tuple) else best_next]
        current_node = current_route[-1]
        battery = best_battery
        time_at = best_time

        # Check if we should close the route (battery low or load high)
        if battery < 30 or current_load > capacity * 0.8:
            if current_route:
                d_ret = dist_matrix[current_node, 0]
                if battery - d_ret >= -1e-9:
                    routes.append(current_route)
                else:
                    for st in instance["stations"]:
                        d_st = dist_matrix[current_node, st]
                        if battery - d_st < -1e-9:
                            continue
                        bat_st = battery - d_st
                        energy_needed = battery_max - bat_st
                        if energy_needed > 0:
                            t_st = time_at + d_st / speed + energy_needed * recharge_rate
                        else:
                            t_st = time_at + d_st / speed
                        bat_st = battery_max
                        d_ret2 = dist_matrix[st, 0]
                        if bat_st - d_ret2 >= -1e-9:
                            routes.append(current_route + [st])
                            break
                    else:
                        routes.append(current_route)
                current_route = []
                current_load = 0
                current_node = 0
                battery = battery_max
                time_at = 0.0

    if current_route:
        routes.append(current_route)

    return routes


def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=80):
    if len(route) < 2:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if is_feasible(new_route):
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break

    return best_route


def _or_opt(route, dist_matrix, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity, max_iters=40):
    """OR-opt: Move segments of length 1, 2 within the route."""
    if len(route) < 3:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        n = len(best_route)
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                segment = best_route[i:i+seg_len]
                for j in range(n - seg_len + 1):
                    if i == j or (j < i + seg_len and j + seg_len > i):
                        continue
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    if j > i:
                        j = j - seg_len
                    new_route = remaining[:j] + segment + remaining[j:]
                    if len(new_route) != len(best_route):
                        continue
                    if is_feasible(new_route):
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break

    return best_route


def _local_search(routes, dist_matrix, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)
    start_time = time.time()

    # Intra-route optimization
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, max_iters=60)
        routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                 stations_set, battery_max, speed, recharge_rate,
                                 horizon, capacity, max_iters=30)
        # Try removing redundant stations
        changed = True
        while changed:
            changed = False
            for pos in range(len(routes[r_idx])):
                if routes[r_idx][pos] in stations_set:
                    trial = routes[r_idx][:pos] + routes[r_idx][pos+1:]
                    if _route_feasible(trial, dist_matrix, instance, n_customers,
                                       stations_set, battery_max, speed, recharge_rate,
                                       horizon, capacity):
                        routes[r_idx] = trial
                        changed = True
                        break
    current_dist = get_total_dist(routes)

    # Inter-route moves
    max_passes = 10
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False

        # Single customer relocate
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Two-customer segment move (OR-opt inter-route)
        if not improved:
            for r1 in range(len(routes)):
                if time.time() > deadline:
                    break
                for seg_len in [2]:
                    for pos1 in range(max(1, len(routes[r1]) - seg_len + 1)):
                        if time.time() > deadline:
                            break
                        segment = routes[r1][pos1:pos1+seg_len]
                        if not all(1 <= c <= n_customers for c in segment):
                            continue
                        seg_load = sum(demands[c] for c in segment)
                        for r2 in range(len(routes)):
                            if time.time() > deadline:
                                break
                            if r1 == r2:
                                continue
                            if route_load(routes[r2]) + seg_load > capacity:
                                continue
                            for pos2 in range(len(routes[r2]) + 1):
                                if time.time() > deadline:
                                    break
                                route1_new = routes[r1][:pos1] + routes[r1][pos1+seg_len:]
                                route2_new = routes[r2][:pos2] + segment + routes[r2][pos2:]
                                if is_feasible(route1_new) and is_feasible(route2_new):
                                    new_routes = [r[:] for r in routes]
                                    new_routes[r1] = route1_new
                                    new_routes[r2] = route2_new
                                    new_dist = get_total_dist(new_routes)
                                    if new_dist < current_dist - 1e-9:
                                        routes = new_routes
                                        current_dist = new_dist
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

    routes = [r for r in routes if len(r) > 0]
    return routes


def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 12.0
    end_temp = 0.02
    max_iters = 5000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp
    no_improve_count = 0
    max_no_improve = 500

    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        if no_improve_count > max_no_improve:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.35:
            # Relocate single customer
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
        elif perturbation_type < 0.55:
            # Swap two customers between routes
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        elif perturbation_type < 0.75:
            # Intra-route 2-opt reversal
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]
        elif perturbation_type < 0.85:
            # Intra-route OR-opt (move segment of length 1 or 2)
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 3:
                    seg_len = random.choice([1, 2])
                    if len(new_routes[r1]) >= seg_len + 1:
                        i = random.randint(0, len(new_routes[r1]) - seg_len)
                        segment = new_routes[r1][i:i+seg_len]
                        remaining = new_routes[r1][:i] + new_routes[r1][i+seg_len:]
                        j = random.randint(0, len(remaining))
                        new_routes[r1] = remaining[:j] + segment + remaining[j:]
        else:
            # Inter-route segment move (2 customers)
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    pos1 = random.randint(0, len(new_routes[r1]) - 2)
                    segment = new_routes[r1][pos1:pos1+2]
                    if all(1 <= c <= n_customers for c in segment):
                        seg_load = sum(demands[c] for c in segment)
                        valid_targets = [r for r in range(len(new_routes))
                                         if r != r1 and route_load(new_routes[r]) + seg_load <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r1] = new_routes[r1][:pos1] + new_routes[r1][pos1+2:]
                            new_routes[r2] = new_routes[r2][:pos2] + segment + new_routes[r2][pos2:]

        new_routes = [r for r in new_routes if len(r) > 0]
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist
                    no_improve_count = 0
                else:
                    no_improve_count += 1
        else:
            no_improve_count += 1

        temp *= cooling_rate

    return best_routes


def _polish_routes(routes, dist_matrix, instance, n_customers, stations_set,
                   battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Final polish: multiple passes of 2-opt + OR-opt + inter-route moves."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)

    for _pass in range(3):
        if time.time() > deadline:
            break
        for r_idx in range(len(routes)):
            if time.time() > deadline:
                break
            routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed, recharge_rate,
                                      horizon, capacity, max_iters=40)
            routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                     stations_set, battery_max, speed, recharge_rate,
                                     horizon, capacity, max_iters=25)

        # Inter-route moves
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - 1e-9:
                                routes = new_routes
                                current_dist = new_dist
                                break
                    if time.time() > deadline:
                        break

    routes = [r for r in routes if len(r) > 0]
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    best_routes = None
    best_dist = float('inf')

    # Phase 1: Generate diverse initial solutions
    # Savings-based constructions
    for i in range(12):
        if time.time() > deadline - 40:
            break
        routes = _savings_construct(instance, dist_matrix, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity,
                                     seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Nearest neighbor constructions
    for i in range(8):
        if time.time() > deadline - 30:
            break
        routes = _nearest_neighbor_construct(instance, dist_matrix, n_customers, stations_set,
                                              battery_max, speed, recharge_rate, horizon, capacity,
                                              seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local search
    if time.time() < deadline - 20:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 20)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 10)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Polish
    if time.time() < deadline - 5:
        polished = _polish_routes(best_routes, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(polished, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = polished

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
