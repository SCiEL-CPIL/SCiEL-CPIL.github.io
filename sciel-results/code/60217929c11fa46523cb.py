# EVOLVE-BLOCK-START
import numpy as np
import math
import time as _time

def _dist_matrix(coords):
    n = len(coords)
    c = np.asarray(coords, dtype=float)
    d = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            v = math.hypot(dx, dy)
            d[i, j] = v
            d[j, i] = v
    return d

def _simulate(route, d, ready, due, service, demand, cap, batt, horizon, speed,
              recharge_rate, stations):
    """Return (feasible, total_dist, end_time)."""
    t = 0.0
    b = float(batt)
    load = 0.0
    dist = 0.0
    prev = 0
    for node in route:
        dist += d[prev, node]
        t += d[prev, node] / speed
        b -= d[prev, node]
        if b < -1e-9:
            return (False, dist, t)
        if node in stations:
            if b < 0.0:
                b = 0.0
            t += (float(batt) - b) * recharge_rate
            b = float(batt)
        else:
            load += demand[node]
            if load > cap + 1e-9:
                return (False, dist, t)
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return (False, dist, t)
            t += service[node]
        prev = node
    # back to depot
    dist += d[prev, 0]
    t += d[prev, 0] / speed
    b -= d[prev, 0]
    if b < -1e-9:
        return (False, dist, t)
    if t > horizon + 1e-9:
        return (False, dist, t)
    return (True, dist, t)

def _route_total_dist(route, d):
    if not route:
        return 0.0
    s = d[0, route[0]]
    for i in range(len(route) - 1):
        s += d[route[i], route[i + 1]]
    s += d[route[-1], 0]
    return s

def _is_feasible(route, d, ready, due, service, demand, cap, batt, horizon, speed,
                 recharge_rate, stations):
    ok, _, _ = _simulate(route, d, ready, due, service, demand, cap, batt,
                         horizon, speed, recharge_rate, stations)
    return ok

def _total_distance(routes, d):
    return sum(_route_total_dist(r, d) for r in routes)

def _construct_savings(customers, d, ready, due, service, demand, cap, batt,
                       horizon, speed, recharge_rate, stations):
    # start with one customer per route
    routes = [[c] for c in customers]
    idx_of = {c: i for i, c in enumerate(customers)}
    n = len(routes)
    # savings for merging route A (ending with a) and route B (starting with b)
    # saving = d[0,a]+d[0,b]-d[a,b]
    # We merge only if both routes are singletons OR we just keep trying.
    # Standard: maintain route ends. For simplicity, use classic savings on
    # current route endpoints.
    # Precompute savings for all pairs of customers.
    sav = {}
    for a in customers:
        for b in customers:
            if a != b:
                sav[(a, b)] = d[0, a] + d[0, b] - d[a, b]
    order = sorted(sav.keys(), key=lambda k: sav[k], reverse=True)
    # route membership
    route_of = {c: [c] for c in customers}  # dict customer -> list route
    active = {c: True for c in customers}
    for (a, b) in order:
        if not active.get(a) or not active.get(b):
            continue
        ra = route_of[a]
        rb = route_of[b]
        if ra is rb:
            continue
        # a must be the last of ra, b must be the first of rb
        if ra[-1] != a or rb[0] != b:
            continue
        merged = ra + rb
        # feasibility check
        if not _is_feasible(merged, d, ready, due, service, demand, cap, batt,
                            horizon, speed, recharge_rate, stations):
            continue
        # commit merge
        for c in ra:
            route_of[c] = merged
            active[c] = True
        for c in rb:
            route_of[c] = merged
        # mark old endpoints inactive as merge points
        active[a] = False  # a no longer an endpoint (it's internal now)
        active[b] = False
        # rebuild routes list
        routes = []
        seen = set()
        for c in customers:
            r = route_of[c]
            key = id(r)
            if key not in seen:
                seen.add(key)
                routes.append(list(r))
    return routes

def _try_relocate(routes, d, ready, due, service, demand, cap, batt, horizon,
                  speed, recharge_rate, stations):
    improved = True
    best = routes
    while improved:
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                c = routes[i][j]
                new_i = routes[i][:j] + routes[i][j + 1:]
                for k in range(len(routes)):
                    if k == i:
                        continue
                    base = routes[k]
                    for pos in range(len(base) + 1):
                        cand = base[:pos] + [c] + base[pos:]
                        if not _is_feasible(new_i, d, ready, due, service,
                                            demand, cap, batt, horizon, speed,
                                            recharge_rate, stations):
                            continue
                        if not _is_feasible(cand, d, ready, due, service,
                                            demand, cap, batt, horizon, speed,
                                            recharge_rate, stations):
                            continue
                        new_routes = [list(x) for x in routes]
                        new_routes[i] = new_i
                        new_routes[k] = cand
                        new_routes = [r for r in new_routes if r]
                        nd = _total_distance(new_routes, d)
                        od = _total_distance(routes, d)
                        if nd < od - 1e-9:
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes

def _try_2opt(route, d, ready, due, service, demand, cap, batt, horizon,
              speed, recharge_rate, stations):
    n = len(route)
    if n < 2:
        return route
    best = route
    best_d = _route_total_dist(route, d)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = route[:i] + list(reversed(route[i:j + 1])) + route[j + 1:]
                if cand == best:
                    continue
                if not _is_feasible(cand, d, ready, due, service, demand, cap,
                                    batt, horizon, speed, recharge_rate,
                                    stations):
                    continue
                cd = _route_total_dist(cand, d)
                if cd < best_d - 1e-9:
                    best = cand
                    best_d = cd
                    improved = True
    return best

def _try_oropt(route, d, ready, due, service, demand, cap, batt, horizon,
               speed, recharge_rate, stations):
    n = len(route)
    if n < 2:
        return route
    best = route
    best_d = _route_total_dist(route, d)
    improved = True
    while improved:
        improved = False
        for L in range(1, min(3, n) + 1):
            for i in range(n - L + 1):
                seg = route[i:i + L]
                rest = route[:i] + route[i + L:]
                for pos in range(len(rest) + 1):
                    cand = rest[:pos] + seg + rest[pos:]
                    if cand == best:
                        continue
                    if not _is_feasible(cand, d, ready, due, service, demand,
                                        cap, batt, horizon, speed,
                                        recharge_rate, stations):
                        continue
                    cd = _route_total_dist(cand, d)
                    if cd < best_d - 1e-9:
                        best = cand
                        best_d = cd
                        improved = True
    return best

def _local_search(routes, d, ready, due, service, demand, cap, batt, horizon,
                  speed, recharge_rate, stations, time_limit):
    t0 = _time.time()
    while _time.time() - t0 < time_limit:
        before = _total_distance(routes, d)
        # 2-opt and or-opt per route
        new_routes = []
        for r in routes:
            r = _try_2opt(r, d, ready, due, service, demand, cap, batt,
                          horizon, speed, recharge_rate, stations)
            r = _try_oropt(r, d, ready, due, service, demand, cap, batt,
                           horizon, speed, recharge_rate, stations)
            new_routes.append(r)
        routes = [r for r in new_routes if r]
        # relocate
        routes = _try_relocate(routes, d, ready, due, service, demand, cap,
                               batt, horizon, speed, recharge_rate, stations)
        after = _total_distance(routes, d)
        if after >= before - 1e-9:
            break
    return routes

def plan(instance):
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demand = instance["demand"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    cap = instance["capacity"]
    batt = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    t_start = _time.time()
    deadline = t_start + 140.0  # safety margin

    d = _dist_matrix(coords)
    customers = list(range(1, n_customers + 1))

    # Construction: savings
    routes = _construct_savings(customers, d, ready, due, service, demand, cap,
                                batt, horizon, speed, recharge_rate, stations)

    # Local search with time budget
    if _time.time() < deadline - 20.0:
        routes = _local_search(routes, d, ready, due, service, demand, cap,
                               batt, horizon, speed, recharge_rate, stations,
                               time_limit=max(5.0, deadline - _time.time() - 5.0))

    # Final validation; if infeasible, fall back to one-customer-per-route
    all_ok = True
    for r in routes:
        if not _is_feasible(r, d, ready, due, service, demand, cap, batt,
                            horizon, speed, recharge_rate, stations):
            all_ok = False
            break
    if not all_ok:
        routes = [[c] for c in customers]

    return routes
# EVOLVE-BLOCK-END
