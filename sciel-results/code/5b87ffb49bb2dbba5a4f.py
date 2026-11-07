# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np

_EPS = 1e-9


def _dist(a, b, coords):
    return math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])


def _check_route(route, coords, demands, ready, due, service, stations,
                 capacity, battery, speed, recharge_rate, horizon):
    """Return (total_distance, valid)."""
    if not route:
        return 0.0, True
    total = 0.0
    t = 0.0
    b = battery
    load = 0.0
    prev = 0
    for node in route:
        d = _dist(prev, node, coords)
        total += d
        t += d / speed
        b -= d
        if b < -_EPS:
            return total, False
        if node in stations:
            # Refill: adds (battery - energy)*recharge_rate time, then refills.
            t += (battery - b) * recharge_rate
            b = battery
        else:
            load += demands[node]
            if load > capacity + _EPS:
                return total, False
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return total, False
            t += service[node]
        prev = node
    # Return to depot.
    d = _dist(prev, 0, coords)
    total += d
    t += d / speed
    b -= d
    if b < -_EPS:
        return total, False
    if t > horizon + _EPS:
        return total, False
    return total, True


def _route_cost(route, coords, demands, ready, due, service, stations,
                capacity, battery, speed, recharge_rate, horizon):
    total, valid = _check_route(route, coords, demands, ready, due, service,
                                stations, capacity, battery, speed,
                                recharge_rate, horizon)
    if not valid:
        return float("inf")
    return total


def _two_opt(route, coords, demands, ready, due, service, stations,
             capacity, battery, speed, recharge_rate, horizon, max_iter=200):
    """2-opt on a single route (no charging), returns improved route."""
    n = len(route)
    if n < 2:
        return route
    cur = _route_cost(route, coords, demands, ready, due, service, stations,
                      capacity, battery, speed, recharge_rate, horizon)
    if not math.isfinite(cur):
        return route
    best = list(route)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                c = _route_cost(cand, coords, demands, ready, due, service,
                                stations, capacity, battery, speed,
                                recharge_rate, horizon)
                if c < cur - 1e-7:
                    cur = c
                    best = cand
                    improved = True
    return best


def _relocate(route, coords, demands, ready, due, service, stations,
              capacity, battery, speed, recharge_rate, horizon):
    """Try moving each customer to a better position in the same route."""
    n = len(route)
    if n < 2:
        return route
    cur = _route_cost(route, coords, demands, ready, due, service, stations,
                      capacity, battery, speed, recharge_rate, horizon)
    if not math.isfinite(cur):
        return route
    best = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(len(best)):
            node = best[i]
            rest = best[:i] + best[i + 1:]
            for k in range(len(rest) + 1):
                if k == i or k == i - 1:
                    continue
                cand = rest[:k] + [node] + rest[k:]
                c = _route_cost(cand, coords, demands, ready, due, service,
                                stations, capacity, battery, speed,
                                recharge_rate, horizon)
                if c < cur - 1e-7:
                    cur = c
                    best = cand
                    improved = True
                    break
    return best


def _build_routes(instance, seed, deadline):
    random.seed(seed)
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_cust + 1))

    # Sort by due time (earliest first) to make insertion feasible.
    customers.sort(key=lambda v: due[v])

    # Greedy nearest-neighbour insertion into routes.
    routes = []
    route_loads = []

    def _nearest_route_cost(cand_route, node):
        return _route_cost(cand_route, coords, demands, ready, due, service,
                           stations, capacity, battery, speed,
                           recharge_rate, horizon)

    for node in customers:
        best_cost = float("inf")
        best_route_idx = -1
        best_pos = -1
        # Try inserting into existing routes at each position.
        for ri, r in enumerate(routes):
            if route_loads[ri] + demands[node] > capacity + _EPS:
                continue
            for pos in range(len(r) + 1):
                cand = r[:pos] + [node] + r[pos:]
                c = _nearest_route_cost(cand, node)
                if c < best_cost:
                    best_cost = c
                    best_route_idx = ri
                    best_pos = pos
        # Try starting a new route.
        new_c = _nearest_route_cost([node], node)
        if new_c < best_cost:
            best_cost = new_c
            best_route_idx = -1
            best_pos = -1

        if best_route_idx == -1:
            routes.append([node])
            route_loads.append(demands[node])
        else:
            r = routes[best_route_idx]
            r.insert(best_pos, node)
            route_loads[best_route_idx] += demands[node]

    # Local search: 2-opt + relocate per route.
    for ri in range(len(routes)):
        routes[ri] = _two_opt(routes[ri], coords, demands, ready, due,
                              service, stations, capacity, battery, speed,
                              recharge_rate, horizon)
        routes[ri] = _relocate(routes[ri], coords, demands, ready, due,
                               service, stations, capacity, battery, speed,
                               recharge_rate, horizon)
        routes[ri] = _two_opt(routes[ri], coords, demands, ready, due,
                              service, stations, capacity, battery, speed,
                              recharge_rate, horizon)

    return routes


def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # safety margin within 150s kill

    # Run multiple seeds, keep the best valid result.
    best_routes = None
    best_cost = float("inf")

    seeds = [42, 1, 7, 123, 2024, 555, 999, 31415]
    for seed in seeds:
        if time.time() > DEADLINE - 5.0:
            break
        try:
            routes = _build_routes(instance, seed, DEADLINE)
            # Validate all routes and check every customer served exactly once.
            total = 0.0
            served = set()
            valid = True
            for r in routes:
                c, ok = _check_route(r, np.array(instance["coords"], dtype=float),
                                     np.array(instance["demands"], dtype=float),
                                     np.array(instance["ready"], dtype=float),
                                     np.array(instance["due"], dtype=float),
                                     np.array(instance["service"], dtype=float),
                                     set(instance["stations"]),
                                     instance["capacity"], instance["battery"],
                                     instance["speed"], instance["recharge_rate"],
                                     instance["horizon"])
                if not ok:
                    valid = False
                    break
                total += c
                for v in r:
                    if v in served:
                        valid = False
                        break
                    served.add(v)
                if not valid:
                    break
            if valid and served == set(range(1, instance["n_customers"] + 1)):
                if total < best_cost:
                    best_cost = total
                    best_routes = routes
        except Exception:
            continue

    if best_routes is None:
        # Fallback: one customer per route (always feasible).
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        best_routes = [[v] for v in customers]

    return best_routes
# EVOLVE-BLOCK-END
