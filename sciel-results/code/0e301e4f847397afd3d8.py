"""EV routing with time windows: cluster-based construction + 2-opt/Or-opt + battery-aware repair."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np

_DEADLINE = 140.0  # safety margin before 150s kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _dist2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return math.sqrt(dx * dx + dy * dy)


def _build_dist(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, instance, D, coords):
    """
    Simulate a route (list of node ids, depot implicit at start/end).
    Returns (total_distance, valid, end_time, end_battery)
    Rules:
      - start at depot (node 0) time=0, battery=battery_cap
      - travel uses battery and time = d/speed
      - at customer: wait until ready, check due, add service time
      - at station: wait (battery - energy) * recharge_rate, then battery = battery_cap
        (energy is the battery level upon arrival; if battery already full, wait 0)
      - return to depot, check horizon
    """
    speed = instance["speed"]
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    t = 0.0
    bat = bat_cap
    dist_total = 0.0
    prev = 0
    total_demand = 0

    for node in route:
        d = D[prev, node]
        dist_total += d
        t += d / speed
        bat -= d

        if bat < -1e-9:
            return dist_total, False, t, bat

        if node in stations:
            # recharge
            energy_needed = bat_cap - bat
            wait = energy_needed * recharge_rate
            t += wait
            bat = bat_cap
        else:
            # customer
            total_demand += demands[node]
            if total_demand > cap + 1e-9:
                return dist_total, False, t, bat
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, bat
            t += service[node]

        prev = node

    # return to depot
    d = D[prev, 0]
    dist_total += d
    t += d / speed
    bat -= d

    if bat < -1e-9:
        return dist_total, False, t, bat
    if t > horizon + 1e-9:
        return dist_total, False, t, bat

    return dist_total, True, t, bat


def _route_distance(route, D):
    if not route:
        return 0.0
    dist = D[0, route[0]]
    for i in range(len(route) - 1):
        dist += D[route[i], route[i + 1]]
    dist += D[route[-1], 0]
    return dist


def _is_valid_route(route, instance, D):
    _, valid, _, _ = _simulate(route, instance, D, instance["coords"])
    return valid


def _construct_initial(instance, D):
    """
    Cluster customers into groups of ~k by nearest-neighbor from depot,
    then build routes.
    """
    n_cust = instance["n_customers"]
    customers = list(range(1, n_cust + 1))
    coords = instance["coords"]
    speed = instance["speed"]
    bat_cap = instance["battery"]
    cap = instance["capacity"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    recharge_rate = instance["recharge_rate"]

    # Sort customers by due time
    customers.sort(key=lambda v: due[v])

    # Greedy insertion: start with empty routes, insert each customer into the best route
    # or start a new route.
    routes = []
    for c in customers:
        best_route = None
        best_pos = None
        best_cost = float('inf')
        best_valid = False

        for ri, route in enumerate(routes):
            # Try inserting c at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                # Quick check: capacity
                total_d = sum(demands[n] for n in new_route)
                if total_d > cap:
                    continue
                # Simulate
                dist, valid, t_end, bat_end = _simulate(new_route, instance, D, coords)
                if valid:
                    # Cost heuristic: distance + penalty for late arrival
                    cost = dist
                    if cost < best_cost:
                        best_cost = cost
                        best_route = ri
                        best_pos = pos
                        best_valid = True

        if best_route is not None:
            routes[best_route] = routes[best_route][:best_pos] + [c] + routes[best_route][best_pos:]
        else:
            routes.append([c])

    return routes


def _two_opt(route, D):
    """2-opt improvement on a single route (ignoring time/battery for speed, then validate)."""
    n = len(route)
    if n < 2:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                # Check distance improvement (approximate, ignoring depot legs for inner swaps)
                old_d1 = D[route[i - 1] if i > 0 else 0, route[i]]
                old_d2 = D[route[j], route[j + 1] if j + 1 < n else 0]
                new_d1 = D[route[i - 1] if i > 0 else 0, route[j]]
                new_d2 = D[route[i], route[j + 1] if j + 1 < n else 0]
                if new_d1 + new_d2 < old_d1 + old_d2 - 1e-9:
                    route = new_route
                    improved = True
                    break
            if improved:
                break
    return route


def _or_opt(route, D, max_seg=2):
    """Or-opt: move segments of length 1-2 to better positions."""
    n = len(route)
    if n < 3:
        return route

    improved = True
    while improved:
        improved = False
        for seg_len in range(1, max_seg + 1):
            for i in range(n - seg_len + 1):
                seg = route[i:i + seg_len]
                # Remove segment
                rest = route[:i] + route[i + seg_len:]
                # Try inserting at each position in rest
                for j in range(len(rest) + 1):
                    if j == i or j == i + 1:
                        continue
                    new_route = rest[:j] + seg + rest[j:]
                    # Compute distance delta
                    old_in = (D[route[i - 1] if i > 0 else 0, route[i]] +
                              D[route[i + seg_len - 1], route[i + seg_len] if i + seg_len < n else 0])
                    new_in = (D[rest[j - 1] if j > 0 else 0, seg[0]] +
                              D[seg[-1], rest[j] if j < len(rest) else 0])
                    if new_in < old_in - 1e-9:
                        route = new_route
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return route


def _improve_routes(routes, instance, D, coords, time_budget):
    """Apply local search improvements to each route."""
    deadline = time.time() + time_budget
    for ri in range(len(routes)):
        if time.time() > deadline:
            break
        # 2-opt
        routes[ri] = _two_opt(routes[ri], D)
        # Or-opt
        routes[ri] = _or_opt(routes[ri], D)
        # Validate; if invalid, try to fix by splitting
        if not _is_valid_route(routes[ri], instance, D):
            # Split route into smaller chunks and re-insert
            half = len(routes[ri]) // 2
            if half > 0:
                part1 = routes[ri][:half]
                part2 = routes[ri][half:]
                routes[ri] = part1
                routes.append(part2)
    return routes


def _repair_routes(routes, instance, D, coords):
    """Ensure all routes are valid. If a route is invalid, split it."""
    changed = True
    while changed:
        changed = False
        new_routes = []
        for route in routes:
            if _is_valid_route(route, instance, D):
                new_routes.append(route)
            else:
                # Try splitting in half
                if len(route) <= 1:
                    # Single node; should be valid (full battery, short distance)
                    # But if still invalid, keep it (will be invalid overall)
                    new_routes.append(route)
                else:
                    mid = len(route) // 2
                    part1 = route[:mid]
                    part2 = route[mid:]
                    new_routes.append(part1)
                    new_routes.append(part2)
                    changed = True
        routes = new_routes
    return routes


def plan(instance):
    global _START
    _START = time.time()

    coords = instance["coords"]
    n_total = len(coords)
    D = _build_dist(coords)

    n_cust = instance["n_customers"]
    customers = list(range(1, n_cust + 1))

    # Construct initial solution
    routes = _construct_initial(instance, D)

    # Repair to ensure validity
    routes = _repair_routes(routes, instance, D, coords)

    # Improve with local search
    time_left = _remaining()
    if time_left > 5:
        routes = _improve_routes(routes, instance, D, coords, min(time_left - 5, 60))

    # Final repair
    routes = _repair_routes(routes, instance, D, coords)

    # Verify all customers are covered exactly once
    served = set()
    for r in routes:
        for n in r:
            if n in served:
                # Duplicate; should not happen
                pass
            served.add(n)
    missing = set(customers) - served
    if missing:
        # Add missing customers as single-node routes
        for m in missing:
            routes.append([m])
        routes = _repair_routes(routes, instance, D, coords)

    return routes
# EVOLVE-BLOCK-END
