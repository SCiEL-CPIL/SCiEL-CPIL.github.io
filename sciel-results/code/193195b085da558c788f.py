"""EV routing with time windows: cluster-first + nearest-neighbor + multi-neighborhood local search."""
# EVOLVE-BLOCK-START
import math
import time
import random

_EPS = 1e-9


def _euclid(coords, i, j):
    xi, yi = coords[i]
    xj, yj = coords[j]
    return math.hypot(xi - xj, yi - yj)


def _build_distance_matrix(coords):
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        for j in range(i + 1, n):
            xj, yj = coords[j]
            d = math.hypot(xi - xj, yi - yj)
            D[i][j] = d
            D[j][i] = d
    return D


def _check_route(route, instance, D, stations_set):
    """Return (ok, total_distance, total_time)."""
    if not route:
        return True, 0.0, 0.0
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    load = 0.0
    battery = float(battery_max)
    t = 0.0
    prev = 0
    total = 0.0

    for node in route:
        d = D[prev][node]
        if d > battery + _EPS:
            return False, total, t
        battery -= d
        t += d / speed
        total += d
        prev = node
        if node in stations_set:
            if battery < 0.0:
                return False, total, t
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            continue
        load += demands[node]
        if load > capacity + _EPS:
            return False, total, t
        if t < ready[node]:
            t = float(ready[node])
        if t > due[node] + _EPS:
            return False, total, t
        t += service[node]
        if t > horizon + _EPS:
            return False, total, t

    d = D[prev][0]
    if d > battery + _EPS:
        return False, total, t
    battery -= d
    t += d / speed
    total += d
    if t > horizon + _EPS:
        return False, total, t
    return True, total, t


def _route_cost(route, D):
    if not route:
        return 0.0
    total = D[0][route[0]]
    for i in range(1, len(route)):
        total += D[route[i - 1]][route[i]]
    total += D[route[-1]][0]
    return total


def _total_cost(routes, D):
    return sum(_route_cost(r, D) for r in routes)


def _nearest_neighbor_construct(instance, D, stations_set, deadline):
    """Build routes using nearest-neighbor with battery/time feasibility."""
    n_cust = instance["n_customers"]
    customers = list(range(1, n_cust + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    remaining = set(customers)
    routes = []

    while remaining and time.time() < deadline:
        # Start a new route from depot
        route = []
        load = 0.0
        battery = float(battery_max)
        t = 0.0
        prev = 0

        while remaining:
            best_node = None
            best_score = float('inf')
            best_battery_after = 0.0
            best_time_after = 0.0

            for c in remaining:
                d = D[prev][c]
                if d > battery + _EPS:
                    continue
                battery_after = battery - d
                t_arrive = t + d / speed
                if t_arrive < ready[c]:
                    t_arrive = float(ready[c])
                if t_arrive > due[c] + _EPS:
                    continue
                t_after = t_arrive + service[c]
                if t_after > horizon + _EPS:
                    continue
                load_after = load + demands[c]
                if load_after > capacity + _EPS:
                    continue
                # Check if we can return to depot after this
                d_back = D[c][0]
                if d_back > battery_after + _EPS:
                    continue
                t_back = t_after + d_back / speed
                if t_back > horizon + _EPS:
                    continue

                score = d
                if score < best_score:
                    best_score = score
                    best_node = c
                    best_battery_after = battery_after
                    best_time_after = t_after

            if best_node is None:
                # Try inserting a station to extend range
                inserted = False
                for s in instance["stations"]:
                    d_to_s = D[prev][s]
                    if d_to_s > battery + _EPS:
                        continue
                    battery_at_s = battery - d_to_s
                    t_at_s = t + d_to_s / speed
                    charge_time = (battery_max - battery_at_s) * recharge
                    t_at_s += charge_time
                    # Now try to go to a customer from station
                    for c in remaining:
                        d_s_c = D[s][c]
                        if d_s_c > battery_max + _EPS:
                            continue
                        battery_after = battery_max - d_s_c
                        t_arrive = t_at_s + d_s_c / speed
                        if t_arrive < ready[c]:
                            t_arrive = float(ready[c])
                        if t_arrive > due[c] + _EPS:
                            continue
                        t_after = t_arrive + service[c]
                        if t_after > horizon + _EPS:
                            continue
                        load_after = load + demands[c]
                        if load_after > capacity + _EPS:
                            continue
                        d_back = D[c][0]
                        if d_back > battery_after + _EPS:
                            continue
                        t_back = t_after + d_back / speed
                        if t_back > horizon + _EPS:
                            continue
                        # Feasible via station
                        route.append(s)
                        route.append(c)
                        load = load_after
                        battery = battery_after
                        t = t_after
                        prev = c
                        remaining.discard(c)
                        inserted = True
                        break
                    if inserted:
                        break
                if not inserted:
                    break

            route.append(best_node)
            load = load + demands[best_node]
            battery = best_battery_after
            t = best_time_after
            prev = best_node
            remaining.discard(best_node)

        if route:
            routes.append(route)

    # Check if any customers are missing and add them as single-node routes
    covered = set()
    for r in routes:
        for node in r:
            if node not in stations_set and node != 0:
                covered.add(node)
    missing = [v for v in range(1, n_cust + 1) if v not in covered]
    for v in missing:
        routes.append([v])

    return routes


def _try_add_station(route, instance, D, stations_set):
    """Try inserting a charging station to make route feasible. Returns fixed route or None."""
    n = len(route)
    best_route = None
    best_dist = float('inf')
    for s in instance["stations"]:
        for p in range(n + 1):
            cand = route[:p] + [s] + route[p:]
            ok, dist, _ = _check_route(cand, instance, D, stations_set)
            if ok and dist < best_dist:
                best_dist = dist
                best_route = cand
    return best_route


def _two_opt(routes, instance, D, stations_set, deadline):
    """2-opt within each route."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            m = len(r)
            if m < 3:
                continue
            for i in range(m - 1):
                if time.time() >= deadline:
                    return routes
                for j in range(i + 1, m):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    ok, _, _ = _check_route(cand, instance, D, stations_set)
                    if ok and _route_cost(cand, D) < _route_cost(r, D) - _EPS:
                        routes[ri] = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def _or_opt(routes, instance, D, stations_set, deadline):
    """Or-opt: move 1, 2, or 3 consecutive nodes within a route or to another route."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(len(routes)):
            r = routes[ri]
            m = len(r)
            if m < 2:
                continue
            for seg_len in (1, 2, 3):
                if m < seg_len + 1:
                    continue
                for i in range(m - seg_len + 1):
                    if time.time() >= deadline:
                        return routes
                    seg = r[i:i + seg_len]
                    if any(n in stations_set for n in seg):
                        continue
                    rest = r[:i] + r[i + seg_len:]
                    old_d = _route_cost(r, D)
                    for j in range(len(rest) + 1):
                        if j == i or j == i + seg_len:
                            continue
                        cand = rest[:j] + seg + rest[j:]
                        ok, _, _ = _check_route(cand, instance, D, stations_set)
                        if ok:
                            new_d = _route_cost(cand, D)
                            if new_d < old_d - _EPS:
                                routes[ri] = cand
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        # Cross-route Or-opt: move segment from route a to route b
        for a in range(len(routes)):
            ra = routes[a]
            m = len(ra)
            if m < 2:
                continue
            for seg_len in (1, 2):
                if m < seg_len + 1:
                    continue
                for i in range(m - seg_len + 1):
                    if time.time() >= deadline:
                        return routes
                    seg = ra[i:i + seg_len]
                    if any(n in stations_set for n in seg):
                        continue
                    rest_a = ra[:i] + ra[i + seg_len:]
                    if not rest_a:
                        continue
                    oka, _, _ = _check_route(rest_a, instance, D, stations_set)
                    if not oka:
                        continue
                    for b in range(len(routes)):
                        if b == a:
                            continue
                        rb = routes[b]
                        for j in range(len(rb) + 1):
                            cand_b = rb[:j] + seg + rb[j:]
                            okb, _, _ = _check_route(cand_b, instance, D, stations_set)
                            if okb:
                                new_total = _route_cost(cand_b, D) + _route_cost(rest_a, D)
                                old_total = _route_cost(ra, D) + _route_cost(rb, D)
                                if new_total < old_total - _EPS:
                                    routes[a] = rest_a
                                    routes[b] = cand_b
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    routes = [r for r in routes if r]
    return routes


def _swap(routes, instance, D, stations_set, deadline):
    """Swap two customers between routes."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(len(routes)):
            ra = routes[a]
            for b in range(a + 1, len(routes)):
                rb = routes[b]
                for i, ca in enumerate(ra):
                    if ca in stations_set:
                        continue
                    for j, cb in enumerate(rb):
                        if cb in stations_set:
                            continue
                        ra_new = ra[:]
                        rb_new = rb[:]
                        ra_new[i] = cb
                        rb_new[j] = ca
                        oka, _, _ = _check_route(ra_new, instance, D, stations_set)
                        if not oka:
                            continue
                        okb, _, _ = _check_route(rb_new, instance, D, stations_set)
                        if not okb:
                            continue
                        new_total = _route_cost(ra_new, D) + _route_cost(rb_new, D)
                        old_total = _route_cost(ra, D) + _route_cost(rb, D)
                        if new_total < old_total - _EPS:
                            routes[a] = ra_new
                            routes[b] = rb_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return routes


def _routes_valid(routes, instance, D, stations_set):
    seen = set()
    for r in routes:
        ok, _, _ = _check_route(r, instance, D, stations_set)
        if not ok:
            return False
        for node in r:
            if node in stations_set or node == 0:
                continue
            if node in seen:
                return False
            seen.add(node)
    n_cust = instance["n_customers"]
    if len(seen) != n_cust:
        return False
    return True


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    stations_set = set(instance["stations"])
    D = _build_distance_matrix(instance["coords"])

    # Nearest-neighbor construction
    routes = _nearest_neighbor_construct(instance, D, stations_set, deadline)

    # Ensure all customers covered
    covered = set()
    for r in routes:
        for node in r:
            if node not in stations_set and node != 0:
                covered.add(node)
    missing = [v for v in range(1, instance["n_customers"] + 1) if v not in covered]
    for v in missing:
        routes.append([v])

    # Local search: alternating 2-opt, Or-opt, and swap
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        routes = _two_opt(routes, instance, D, stations_set, time.time() + remaining_time * 0.3)
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        routes = _or_opt(routes, instance, D, stations_set, time.time() + remaining_time * 0.3)
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        routes = _swap(routes, instance, D, stations_set, time.time() + remaining_time * 0.2)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        routes = _two_opt(routes, instance, D, stations_set, time.time() + remaining_time * 0.5)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        routes = _or_opt(routes, instance, D, stations_set, deadline)

    # Final validity check
    if not _routes_valid(routes, instance, D, stations_set):
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    return routes
# EVOLVE-BLOCK-END
