"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _simulate(route, coords, demands, ready, due, service, stations,
              capacity, battery_cap, speed, recharge_rate, horizon):
    """Simulate a route; return (total_distance, valid, end_time, end_battery)."""
    n = len(coords)
    total_dist = 0.0
    t = 0.0
    batt = battery_cap
    load = 0
    prev = 0  # depot

    for node in route:
        d = _euclid(coords[prev], coords[node])
        total_dist += d
        batt -= d
        t += d / speed
        if batt < -1e-9:
            return total_dist, False, t, batt
        load += demands[node]
        if load > capacity:
            return total_dist, False, t, batt
        if node in stations:
            # charge to full
            if batt < 0:
                batt = 0.0
            energy = battery_cap - batt
            t += energy * recharge_rate
            batt = battery_cap
        else:
            # customer: time window + service
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False, t, batt
            t += service[node]
        prev = node

    # return to depot
    d = _euclid(coords[prev], coords[0])
    total_dist += d
    batt -= d
    t += d / speed
    if batt < -1e-9:
        return total_dist, False, t, batt
    if t > horizon + 1e-9:
        return total_dist, False, t, batt
    return total_dist, True, t, batt


def _route_distance(route, coords):
    """Total Euclidean distance including depot legs."""
    if not route:
        return 0.0
    total = 0.0
    prev = 0
    for node in route:
        total += _euclid(coords[prev], coords[node])
        prev = node
    total += _euclid(coords[prev], coords[0])
    return total


def _greedy_insertion(customers, coords, demands, ready, due, service,
                     stations, capacity, battery_cap, speed, recharge_rate,
                     horizon, max_routes=8):
    """Build routes by inserting customers one by one into best feasible position."""
    n_cust = len(customers)
    if n_cust == 0:
        return []

    # Precompute pairwise distances
    node_ids = [0] + list(range(1, 26)) + list(stations)
    # Actually we need all nodes that can appear: depot, customers, stations
    all_nodes = set([0])
    all_nodes.update(range(1, 26))
    all_nodes.update(stations)
    all_nodes = sorted(all_nodes)
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    m = len(all_nodes)
    dist = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            d = _euclid(coords[all_nodes[i]], coords[all_nodes[j]])
            dist[i, j] = d
            dist[j, i] = d

    def dist_between(a, b):
        return dist[node_to_idx[a], node_to_idx[b]]

    # Start with empty routes
    routes = [[]]
    # We'll insert customers in order of due time
    cust_sorted = sorted(customers, key=lambda v: due[v])

    for cust in cust_sorted:
        best_cost = float('inf')
        best_route_idx = -1
        best_pos = -1

        for ri, route in enumerate(routes):
            # Try inserting cust at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                # Check capacity
                load = sum(demands[n] for n in new_route)
                if load > capacity:
                    continue
                # Check battery and time windows via simulation
                total_dist, valid, end_t, end_batt = _simulate(
                    new_route, coords, demands, ready, due, service,
                    stations, capacity, battery_cap, speed, recharge_rate,
                    horizon
                )
                if not valid:
                    continue
                # Cost: prefer shorter routes, fewer routes
                cost = total_dist
                # Small penalty for using more routes
                cost += 0.01 * len(routes)
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = ri
                    best_pos = pos

        if best_route_idx >= 0:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
        else:
            # Can't insert anywhere; start a new route (shouldn't happen often)
            routes.append([cust])

    return [r for r in routes if r]


def _local_search(routes, coords, demands, ready, due, service,
                  stations, capacity, battery_cap, speed, recharge_rate,
                  horizon, time_limit=60.0):
    """2-opt and relocate local search on routes."""
    start = time.time()
    improved = True
    total_dist = sum(_route_distance(r, coords) for r in routes)

    while improved and (time.time() - start) < time_limit:
        improved = False

        # Try 2-opt within each route
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            if n < 3:
                continue
            for i in range(n - 1):
                if (time.time() - start) > time_limit:
                    return routes
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    # Check capacity
                    load = sum(demands[node] for node in new_route)
                    if load > capacity:
                        continue
                    total_d, valid, _, _ = _simulate(
                        new_route, coords, demands, ready, due, service,
                        stations, capacity, battery_cap, speed, recharge_rate,
                        horizon
                    )
                    if not valid:
                        continue
                    # Compute distance change
                    old_d = _route_distance(route, coords)
                    if total_d < old_d - 1e-9:
                        routes[ri] = new_route
                        improved = True

        # Try relocate: move a customer from one route to another
        for ri in range(len(routes)):
            for pos in range(len(routes[ri])):
                if (time.time() - start) > time_limit:
                    return routes
                cust = routes[ri][pos]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos2 in range(len(routes[rj]) + 1):
                        # Remove cust from ri
                        new_ri = routes[ri][:pos] + routes[ri][pos + 1:]
                        # Insert cust into rj
                        new_rj = routes[rj][:pos2] + [cust] + routes[rj][pos2:]
                        # Check capacities
                        load_ri = sum(demands[node] for node in new_ri)
                        load_rj = sum(demands[node] for node in new_rj)
                        if load_ri > capacity or load_rj > capacity:
                            continue
                        # Simulate both
                        valid_ri = True
                        valid_rj = True
                        dist_ri = 0.0
                        dist_rj = 0.0
                        if new_ri:
                            d, v, _, _ = _simulate(
                                new_ri, coords, demands, ready, due, service,
                                stations, capacity, battery_cap, speed, recharge_rate,
                                horizon
                            )
                            if not v:
                                continue
                            dist_ri = d
                        if new_rj:
                            d, v, _, _ = _simulate(
                                new_rj, coords, demands, ready, due, service,
                                stations, capacity, battery_cap, speed, recharge_rate,
                                horizon
                            )
                            if not v:
                                continue
                            dist_rj = d
                        new_total = dist_ri + dist_rj
                        old_total = _route_distance(routes[ri], coords) + _route_distance(routes[rj], coords)
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

    # Deadline guard: total budget ~140s (leave 10s margin)
    deadline = time.time() + 140.0

    # Phase 1: Greedy insertion
    routes = _greedy_insertion(
        customers, coords, demands, ready, due, service,
        stations, capacity, battery_cap, speed, recharge_rate,
        horizon
    )

    # Validate initial solution
    total_dist = 0.0
    for r in routes:
        d, valid, _, _ = _simulate(
            r, coords, demands, ready, due, service,
            stations, capacity, battery_cap, speed, recharge_rate,
            horizon
        )
        if not valid:
            # Fallback: one customer per route
            routes = [[v] for v in customers]
            break
        total_dist += d

    # Phase 2: Local search with time limit
    if time.time() < deadline - 5.0:
        routes = _local_search(
            routes, coords, demands, ready, due, service,
            stations, capacity, battery_cap, speed, recharge_rate,
            horizon, time_limit=max(1.0, deadline - time.time() - 5.0)
        )

    # Final validation
    all_valid = True
    total_dist = 0.0
    for r in routes:
        d, valid, _, _ = _simulate(
            r, coords, demands, ready, due, service,
            stations, capacity, battery_cap, speed, recharge_rate,
            horizon
        )
        if not valid:
            all_valid = False
            break
        total_dist += d

    if not all_valid:
        # Fallback to safe solution
        routes = [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
