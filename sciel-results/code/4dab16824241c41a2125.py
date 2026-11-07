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


def _route_cost_fast(route, coords, demands, ready, due, service, stations_set,
                     capacity, battery, speed, recharge_rate, horizon):
    """Faster route cost using precomputed arrays."""
    if not route:
        return 0.0, True
    total = 0.0
    t = 0.0
    b = battery
    load = 0.0
    prev = 0
    for node in route:
        d = math.hypot(coords[prev][0] - coords[node][0], coords[prev][1] - coords[node][1])
        total += d
        t += d
        b -= d
        if b < -_EPS:
            return total, False
        if node in stations_set:
            t += (battery - b) * recharge_rate
            b = battery
        else:
            load += demands[node]
            if load > capacity + _EPS:
                return total, False
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + _EPS:
                return total, False
            t += service[node]
        prev = node
    d = math.hypot(coords[prev][0] - coords[0][0], coords[prev][1] - coords[0][1])
    total += d
    t += d
    b -= d
    if b < -_EPS:
        return total, False
    if t > horizon + _EPS:
        return total, False
    return total, True


def _sweep_split(route, coords, demands, ready, due, service, stations,
                 capacity, battery, speed, recharge_rate, horizon):
    """Try to split a route into two using a sweep angle."""
    n = len(route)
    if n < 4:
        return [route]
    # Compute centroid
    cx = sum(coords[v][0] for v in route) / n
    cy = sum(coords[v][1] for v in route) / n
    # Sort by angle
    route_sorted = sorted(route, key=lambda v: math.atan2(coords[v][1] - cy, coords[v][0] - cx))
    # Try different split points
    best_split = None
    best_cost = float("inf")
    for k in range(2, n - 1):
        r1 = route_sorted[:k]
        r2 = route_sorted[k:]
        # Check capacity
        load1 = sum(demands[v] for v in r1)
        load2 = sum(demands[v] for v in r2)
        if load1 > capacity + _EPS or load2 > capacity + _EPS:
            continue
        c1 = _route_cost(r1, coords, demands, ready, due, service, stations,
                         capacity, battery, speed, recharge_rate, horizon)
        c2 = _route_cost(r2, coords, demands, ready, due, service, stations,
                         capacity, battery, speed, recharge_rate, horizon)
        if c1 < float("inf") and c2 < float("inf"):
            total = c1 + c2
            if total < best_cost:
                best_cost = total
                best_split = (r1, r2)
    if best_split:
        return [best_split[0], best_split[1]]
    return [route]


def _two_opt(route, coords, demands, ready, due, service, stations,
             capacity, battery, speed, recharge_rate, horizon, max_iter=50):
    """2-opt on a single route."""
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
              capacity, battery, speed, recharge_rate, horizon, max_iter=30):
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
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
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


def _cross_exchange(routes, coords, demands, ready, due, service, stations,
                    capacity, battery, speed, recharge_rate, horizon):
    """Try exchanging segments between two routes."""
    n_routes = len(routes)
    if n_routes < 2:
        return routes
    improved = True
    iters = 0
    while improved and iters < 20:
        improved = False
        iters += 1
        for i in range(n_routes):
            for j in range(i + 1, n_routes):
                r1 = routes[i]
                r2 = routes[j]
                if len(r1) < 2 or len(r2) < 2:
                    continue
                # Try exchanging one customer from each route
                for a in range(len(r1)):
                    for b in range(len(r2)):
                        node1 = r1[a]
                        node2 = r2[b]
                        # Check capacity
                        new_r1 = r1[:a] + [node2] + r1[a + 1:]
                        new_r2 = r2[:b] + [node1] + r2[b + 1:]
                        load1 = sum(demands[v] for v in new_r1)
                        load2 = sum(demands[v] for v in new_r2)
                        if load1 > capacity + _EPS or load2 > capacity + _EPS:
                            continue
                        c1 = _route_cost(new_r1, coords, demands, ready, due, service,
                                         stations, capacity, battery, speed,
                                         recharge_rate, horizon)
                        c2 = _route_cost(new_r2, coords, demands, ready, due, service,
                                         stations, capacity, battery, speed,
                                         recharge_rate, horizon)
                        if c1 < float("inf") and c2 < float("inf"):
                            old_c1 = _route_cost(r1, coords, demands, ready, due, service,
                                                  stations, capacity, battery, speed,
                                                  recharge_rate, horizon)
                            old_c2 = _route_cost(r2, coords, demands, ready, due, service,
                                                  stations, capacity, battery, speed,
                                                  recharge_rate, horizon)
                            if c1 + c2 < old_c1 + old_c2 - 1e-7:
                                routes[i] = new_r1
                                routes[j] = new_r2
                                improved = True
    return routes


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
                c = _route_cost(cand, coords, demands, ready, due, service,
                                stations, capacity, battery, speed,
                                recharge_rate, horizon)
                if c < best_cost:
                    best_cost = c
                    best_route_idx = ri
                    best_pos = pos
        # Try starting a new route.
        new_c = _route_cost([node], coords, demands, ready, due, service,
                            stations, capacity, battery, speed,
                            recharge_rate, horizon)
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

    # Cross-exchange between routes
    routes = _cross_exchange(routes, coords, demands, ready, due, service,
                             stations, capacity, battery, speed,
                             recharge_rate, horizon)

    # Final pass of local search
    for ri in range(len(routes)):
        routes[ri] = _two_opt(routes[ri], coords, demands, ready, due,
                              service, stations, capacity, battery, speed,
                              recharge_rate, horizon)

    return routes


def _try_station_insertion(route, coords, demands, ready, due, service, stations,
                           capacity, battery, speed, recharge_rate, horizon):
    """Try inserting charging stations into the route if needed."""
    # Check if the route is valid without stations
    cost, valid = _check_route(route, coords, demands, ready, due, service,
                               stations, capacity, battery, speed,
                               recharge_rate, horizon)
    if valid:
        return route

    # Try inserting stations at various positions
    best_route = None
    best_cost = float("inf")
    station_list = list(stations)

    # Try single station insertions
    for st in station_list:
        for pos in range(len(route) + 1):
            cand = route[:pos] + [st] + route[pos:]
            c, ok = _check_route(cand, coords, demands, ready, due, service,
                                 stations, capacity, battery, speed,
                                 recharge_rate, horizon)
            if ok and c < best_cost:
                best_cost = c
                best_route = cand

    if best_route is not None:
        return best_route
    return route


def plan(instance):
    t0 = time.time()
    DEADLINE = 140.0  # safety margin within 150s kill

    # Run multiple seeds, keep the best valid result.
    best_routes = None
    best_cost = float("inf")

    seeds = [42, 1, 7, 123, 2024, 555, 999, 31415, 777, 1234, 5678, 4321]
    for seed in seeds:
        if time.time() > DEADLINE - 10.0:
            break
        try:
            routes = _build_routes(instance, seed, DEADLINE)

            # Try station insertion on any invalid routes
            for ri in range(len(routes)):
                routes[ri] = _try_station_insertion(
                    routes[ri], np.array(instance["coords"], dtype=float),
                    np.array(instance["demands"], dtype=float),
                    np.array(instance["ready"], dtype=float),
                    np.array(instance["due"], dtype=float),
                    np.array(instance["service"], dtype=float),
                    set(instance["stations"]),
                    instance["capacity"], instance["battery"],
                    instance["speed"], instance["recharge_rate"],
                    instance["horizon"])

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
