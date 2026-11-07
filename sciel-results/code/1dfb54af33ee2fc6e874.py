"""EV routing with time windows – Clarke-Wright savings + fast local search."""
# EVOLVE-BLOCK-START
import math
import time
import random
import numpy as np


def _build_dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        xi, yi = coords[i][0], coords[i][1]
        for j in range(i + 1, n):
            d = math.hypot(xi - coords[j][0], yi - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate(route, D, instance):
    """Returns (total_dist, valid, end_time, end_battery)."""
    coords_n = len(D)
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    n_customers = instance["n_customers"]

    t = 0.0
    bat = battery_max
    load = 0.0
    total_d = 0.0
    prev = 0

    for node in route:
        d = D[prev][node]
        t += d / speed
        bat -= d
        total_d += d

        if bat < -1e-9:
            return total_d, False, t, bat

        if node in stations:
            energy = battery_max - bat
            t += energy * recharge_rate
            bat = battery_max
        else:
            if node > n_customers:
                return total_d, False, t, bat
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_d, False, t, bat
            t += service[node]
            load += demands[node]
            if load > capacity + 1e-9:
                return total_d, False, t, bat
        prev = node

    d = D[prev][0]
    t += d / speed
    bat -= d
    total_d += d

    if bat < -1e-9 or t > horizon + 1e-9:
        return total_d, False, t, bat

    return total_d, True, t, bat


def _total_dist(routes, D):
    total = 0.0
    for route in routes:
        prev = 0
        for node in route:
            total += D[prev][node]
            prev = node
        total += D[prev][0]
    return total


def _is_valid(routes, D, instance):
    n_customers = instance["n_customers"]
    served = set()
    for route in routes:
        for node in route:
            if node > n_customers or node == 0:
                return False
            if node in served:
                return False
            served.add(node)
    if len(served) != n_customers:
        return False
    for route in routes:
        _, v, _, _ = _simulate(route, D, instance)
        if not v:
            return False
    return True


def _clarke_wright(instance, D):
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    capacity = instance["capacity"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["stations"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Start with each customer in its own route
    routes = [[i] for i in range(1, n_customers + 1)]
    route_of = {i: idx for idx, i in enumerate(range(1, n_customers + 1))}

    # Compute savings
    savings = []
    for i in range(1, n_customers + 1):
        for j in range(i + 1, n_customers + 1):
            s = D[0][i] + D[0][j] - D[i][j]
            if s > 0:
                savings.append((-s, i, j))
    savings.sort()

    for neg_s, i, j in savings:
        if time.time() > 145:
            break
        ri = route_of.get(i)
        rj = route_of.get(j)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Check if i and j are at the ends of their routes
        # i must be at start or end of route_i, j at start or end of route_j
        i_at_start = route_i[0] == i
        i_at_end = route_i[-1] == i
        j_at_start = route_j[0] == j
        j_at_end = route_j[-1] == j

        if not (i_at_start or i_at_end) or not (j_at_start or j_at_end):
            continue

        # Try all 4 combinations of joining
        merged = None
        if i_at_start and j_at_end:
            merged = route_i + route_j
        elif i_at_end and j_at_start:
            merged = route_i + route_j
        elif i_at_start and j_at_start:
            merged = route_i + route_j[::-1]
        elif i_at_end and j_at_end:
            merged = route_i + route_j[::-1]
        else:
            continue

        # Check feasibility of merged route
        # Capacity
        total_demand = sum(demands[c] for c in merged)
        if total_demand > capacity + 1e-9:
            continue

        # Battery and time window check via simulation
        _, valid, _, _ = _simulate(merged, D, instance)
        if not valid:
            continue

        # Merge
        routes[ri] = merged
        for c in route_j:
            route_of[c] = ri
        routes[rj] = []
        # Remove empty route
        routes[rj] = None

    # Clean up
    routes = [r for r in routes if r is not None]
    return routes


def _fix_battery(route, D, instance):
    """Insert charging stations where battery would go negative."""
    stations = instance["stations"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]

    t = 0.0
    bat = battery_max
    prev = 0
    new_route = []

    for node in route:
        d = D[prev][node]
        if bat - d < -1e-9:
            # Need to charge before going to node
            # Find reachable station
            best_st = None
            best_d = float('inf')
            for st in stations:
                ds = D[prev][st]
                if bat - ds >= -1e-9 and ds < best_d:
                    best_d = ds
                    best_st = st
            if best_st is None:
                return None
            new_route.append(best_st)
            bat -= best_d
            t += best_d / speed
            energy = battery_max - bat
            t += energy * recharge_rate
            bat = battery_max
            prev = best_st

        new_route.append(node)
        bat -= d
        t += d / speed
        prev = node

    # Check return
    d = D[prev][0]
    if bat - d < -1e-9:
        best_st = None
        best_d = float('inf')
        for st in stations:
            ds = D[prev][st]
            if bat - ds >= -1e-9 and ds < best_d:
                best_d = ds
                best_st = st
        if best_st is None:
            return None
        new_route.append(best_st)

    return new_route


def _local_search(routes, D, instance, time_limit):
    """2-opt + Or-opt + cross-route move."""
    n_customers = instance["n_customers"]
    best = [list(r) for r in routes]
    best_d = _total_dist(best, D)

    def try_2opt(ri):
        nonlocal best, best_d
        route = best[ri]
        n = len(route)
        if n < 2:
            return False
        improved = False
        for i in range(n - 1):
            if time.time() > time_limit:
                return improved
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                _, v, _, _ = _simulate(new_route, D, instance)
                if not v:
                    continue
                nd = best_d - (D[route[i - 1 if i > 0 else 0]][route[i]] +
                               D[route[j]][route[j + 1 if j + 1 < n else 0]])
                # Recompute properly
                old_seg = D[0 if i == 0 else route[i - 1]][route[i]] + D[route[j]][0 if j == n - 1 else route[j + 1]]
                new_seg = D[0 if i == 0 else route[i - 1]][route[j]] + D[route[i]][0 if j == n - 1 else route[j + 1]]
                delta = new_seg - old_seg
                if delta < -1e-9:
                    best[ri] = new_route
                    best_d += delta
                    improved = True
        return improved

    def try_or_opt(ri):
        nonlocal best, best_d
        route = best[ri]
        n = len(route)
        if n < 3:
            return False
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                if time.time() > time_limit:
                    return improved
                seg = route[i:i + seg_len]
                rest = route[:i] + route[i + seg_len:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    new_route = rest[:j] + seg + rest[j:]
                    if new_route == route:
                        continue
                    _, v, _, _ = _simulate(new_route, D, instance)
                    if not v:
                        continue
                    nd = _total_dist([new_route if idx == ri else best[idx] for idx in range(len(best))], D)
                    if nd < best_d - 1e-9:
                        best[ri] = new_route
                        best_d = nd
                        improved = True
                        break
        return improved

    def try_cross_move():
        nonlocal best, best_d
        improved = False
        for ri in range(len(best)):
            for rj in range(ri + 1, len(best)):
                if time.time() > time_limit:
                    return improved
                route_i = best[ri]
                route_j = best[rj]
                for i in range(len(route_i)):
                    for j in range(len(route_j)):
                        a = route_i[i]
                        b = route_j[j]
                        new_ri = list(route_i)
                        new_rj = list(route_j)
                        new_ri[i] = b
                        new_rj[j] = a
                        _, v1, _, _ = _simulate(new_ri, D, instance)
                        if not v1:
                            continue
                        _, v2, _, _ = _simulate(new_rj, D, instance)
                        if not v2:
                            continue
                        nd = _total_dist([new_ri if idx == ri else (new_rj if idx == rj else best[idx]) for idx in range(len(best))], D)
                        if nd < best_d - 1e-9:
                            best[ri] = new_ri
                            best[rj] = new_rj
                            best_d = nd
                            improved = True
        return improved

    max_rounds = 30
    for rnd in range(max_rounds):
        if time.time() > time_limit:
            break
        improved = False
        for ri in range(len(best)):
            if time.time() > time_limit:
                break
            if try_2opt(ri):
                improved = True
        if time.time() > time_limit:
            break
        for ri in range(len(best)):
            if time.time() > time_limit:
                break
            if try_or_opt(ri):
                improved = True
        if time.time() > time_limit:
            break
        if try_cross_move():
            improved = True
        if not improved:
            break

    return best, best_d


def plan(instance):
    start = time.time()
    deadline = start + 140

    D = _build_dist_matrix(instance["coords"])
    n_customers = instance["n_customers"]

    # Generate initial solution using Clarke-Wright
    routes = _clarke_wright(instance, D)

    # Fix battery issues
    fixed_routes = []
    for r in routes:
        fr = _fix_battery(r, D, instance)
        if fr is not None:
            fixed_routes.append(fr)
        else:
            # Fallback: try with stations inserted more aggressively
            fr = list(r)
            fixed_routes.append(fr)

    # Validate
    if not _is_valid(fixed_routes, D, instance):
        # Fallback: one customer per route
        fixed_routes = [[i] for i in range(1, n_customers + 1)]

    # Local search
    fixed_routes, _ = _local_search(fixed_routes, D, instance, deadline)

    # Final validation
    if not _is_valid(fixed_routes, D, instance):
        fixed_routes = [[i] for i in range(1, n_customers + 1)]

    return fixed_routes
# EVOLVE-BLOCK-END
