"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 125.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)
    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True

    # Precompute per-arrival "arrival time" = t + d/speed, and battery check
    # We handle stations: a station arrival adds recharge time then resets battery.

    def check_route(route):
        """Full feasibility + distance. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0
        if sum(demands[v] for v in route) > capacity + 1e-9:
            return False, 0.0
        t = 0.0
        bat = battery
        prev = 0
        total = 0.0
        for v in route:
            d = dist[prev, v]
            bat -= d
            if bat < -1e-9:
                return False, 0.0
            t += d / speed
            total += d
            if is_station[v]:
                t += bat * recharge_rate
                bat = battery
                prev = v
                continue
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v
        d = dist[prev, 0]
        bat -= d
        if bat < -1e-9:
            return False, 0.0
        t += d / speed
        total += d
        if t > horizon + 1e-9:
            return False, 0.0
        return True, total

    def route_distance(route):
        if not route:
            return 0.0
        total = dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], 0]
        return total

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def validate(routes):
        seen = [False] * n_nodes
        for r in routes:
            f, _ = check_route(r)
            if not f:
                return False
            for v in r:
                if seen[v]:
                    return False
                seen[v] = True
        for v in customers:
            if not seen[v]:
                return False
        return True

    # ---------- Construction ----------
    def savings_construct():
        routes = [[v] for v in customers]
        route_of = {v: i for i, v in enumerate(customers)}

        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                if s > 0:
                    savings.append((s, vi, vj))
        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of.get(vi)
            rj = route_of.get(vj)
            if ri is None or rj is None or ri == rj:
                continue
            route_i = routes[ri]
            route_j = routes[rj]

            options = []
            if route_i[-1] == vi and route_j[0] == vj:
                options.append(route_i + route_j)
            if route_i[0] == vi and route_j[-1] == vj:
                options.append(route_i + route_j[::-1])
            if route_i[-1] == vi and route_j[-1] == vj:
                options.append(route_i[::-1] + route_j)
            if route_i[0] == vi and route_j[0] == vj:
                options.append(route_i[::-1] + route_j[::-1])

            for cand in options:
                if sum(demands[x] for x in cand) > capacity + 1e-9:
                    continue
                f, _ = check_route(cand)
                if f:
                    routes[ri] = cand
                    routes[rj] = []
                    for v in cand:
                        route_of[v] = ri
                    non_empty = [r for r in routes if r]
                    for idx, r in enumerate(non_empty):
                        for v in r:
                            route_of[v] = idx
                    routes = non_empty
                    break
        return [r for r in routes if r]

    def greedy_construct(order):
        remaining = list(order)
        routes = []
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0
            while remaining:
                best = None
                best_score = float('inf')
                best_idx = -1
                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    if bat_after < -1e-9:
                        continue
                    if load + demands[v] > capacity + 1e-9:
                        continue
                    t_after = t + d / speed
                    if is_station[v]:
                        t_after += bat_after * recharge_rate
                        # stations don't have time windows; but we don't serve them as customers
                        # skip stations in greedy (they are not customers)
                        continue
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v] + 1e-9:
                        continue
                    slack = due[v] - t_after
                    score = d - 0.015 * slack
                    if score < best_score:
                        best_score = score
                        best = v
                        best_idx = idx
                if best is None:
                    break
                v = best
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]
                if t < ready[v]:
                    t = ready[v]
                t += service[v]
                route.append(v)
                remaining.pop(best_idx)
                prev = v
            if route:
                routes.append(route)
        return routes

    # ---------- Local search ----------
    def two_opt(route):
        if len(route) < 2:
            return route
        improved = True
        while improved:
            improved = False
            n = len(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    return route
                for b in range(a + 1, n):
                    old = 0.0
                    if a > 0:
                        old += dist[route[a - 1], route[a]]
                    else:
                        old += dist[0, route[a]]
                    if b < n - 1:
                        old += dist[route[b], route[b + 1]]
                    else:
                        old += dist[route[b], 0]
                    new = 0.0
                    if a > 0:
                        new += dist[route[a - 1], route[b]]
                    else:
                        new += dist[0, route[b]]
                    if b < n - 1:
                        new += dist[route[a], route[b + 1]]
                    else:
                        new += dist[route[a], 0]
                    if new < old - 1e-6:
                        nr = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        f, _ = check_route(nr)
                        if f:
                            route = nr
                            improved = True
                            break
                if improved:
                    break
        return route

    def relocate(routes):
        routes = [list(r) for r in routes]
        for _ in range(6):
            if time.time() > deadline:
                break
            improved = False
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                n = len(routes[i])
                for seg_len in (1, 2, 3):
                    if time.time() > deadline:
                        break
                    for j in range(n - seg_len + 1):
                        if time.time() > deadline:
                            break
                        seg = routes[i][j:j + seg_len]
                        rest = routes[i][:j] + routes[i][j + seg_len:]
                        if not rest:
                            continue  # don't empty a route in relocate for simplicity
                        f0, _ = check_route(rest)
                        if not f0:
                            continue
                        # delta of removing seg from route i
                        old_i = 0.0
                        if j > 0:
                            old_i += dist[routes[i][j - 1], routes[i][j]]
                        else:
                            old_i += dist[0, routes[i][j]]
                        if j + seg_len < n:
                            old_i += dist[routes[i][j + seg_len - 1], routes[i][j + seg_len]]
                        else:
                            old_i += dist[routes[i][j + seg_len - 1], 0]
                        new_i = 0.0
                        if j > 0:
                            new_i += dist[rest[j - 1], rest[j]] if j < len(rest) else 0
                        if j < len(rest):
                            pass
                        # simpler: compute new connection in rest
                        if j > 0 and j < len(rest):
                            new_i = dist[rest[j - 1], rest[j]]
                        elif j == 0 and len(rest) > 0:
                            new_i = dist[0, rest[0]]
                        elif j == len(rest) and len(rest) > 0:
                            new_i = dist[rest[-1], 0]
                        elif j == 0 and len(rest) == 0:
                            new_i = 0.0
                        delta_rem = old_i - new_i

                        best_delta = None
                        best_k = -1
                        best_pos = -1
                        for k in range(len(routes)):
                            if time.time() > deadline:
                                break
                            rk = routes[k]
                            for pos in range(len(rk) + 1):
                                # delta adding seg to route k at pos
                                old_k = 0.0
                                if pos > 0 and pos < len(rk):
                                    old_k = dist[rk[pos - 1], rk[pos]]
                                elif pos == 0 and len(rk) > 0:
                                    old_k = dist[0, rk[0]]
                                elif pos == len(rk) and len(rk) > 0:
                                    old_k = dist[rk[-1], 0]
                                new_k = 0.0
                                if pos > 0:
                                    new_k += dist[rk[pos - 1], seg[0]]
                                else:
                                    new_k += dist[0, seg[0]]
                                if pos < len(rk):
                                    new_k += dist[seg[-1], rk[pos]]
                                else:
                                    new_k += dist[seg[-1], 0]
                                delta_add = new_k - old_k
                                delta = delta_rem + delta_add
                                if best_delta is None or delta < best_delta:
                                    best_delta = delta
                                    best_k = k
                                    best_pos = pos
                        if best_delta is None or best_delta >= -1e-6:
                            continue
                        # apply
                        new_routes = [list(r) for r in routes]
                        new_routes[i] = rest
                        rk = new_routes[best_k]
                        new_routes[best_k] = rk[:best_pos] + seg + rk[best_pos:]
                        # verify both affected routes
                        f1, _ = check_route(new_routes[i])
                        f2, _ = check_route(new_routes[best_k])
                        if f1 and f2:
                            routes = new_routes
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    def swap(routes):
        routes = [list(r) for r in routes]
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline:
                        break
                    for a in range(len(routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(routes[j])):
                            vi = routes[i][a]
                            vj = routes[j][b]
                            if vi == vj:
                                continue
                            ri = routes[i]
                            rj = routes[j]
                            old = 0.0
                            if a > 0:
                                old += dist[ri[a - 1], vi]
                            else:
                                old += dist[0, vi]
                            if a < len(ri) - 1:
                                old += dist[vi, ri[a + 1]]
                            else:
                                old += dist[vi, 0]
                            if b > 0:
                                old += dist[rj[b - 1], vj]
                            else:
                                old += dist[0, vj]
                            if b < len(rj) - 1:
                                old += dist[vj, rj[b + 1]]
                            else:
                                old += dist[vj, 0]
                            new = 0.0
                            if a > 0:
                                new += dist[ri[a - 1], vj]
                            else:
                                new += dist[0, vj]
                            if a < len(ri) - 1:
                                new += dist[vj, ri[a + 1]]
                            else:
                                new += dist[vj, 0]
                            if b > 0:
                                new += dist[rj[b - 1], vi]
                            else:
                                new += dist[0, vi]
                            if b < len(rj) - 1:
                                new += dist[vi, rj[b + 1]]
                            else:
                                new += dist[vi, 0]
                            if new >= old - 1e-6:
                                continue
                            nr = [list(r) for r in routes]
                            nr[i][a] = vj
                            nr[j][b] = vi
                            f1, _ = check_route(nr[i])
                            f2, _ = check_route(nr[j])
                            if f1 and f2:
                                routes = nr
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ---------- Main ----------
    best_routes = [[v] for v in customers]
    best_d = total_distance(best_routes)

    cands = []
    sw = savings_construct()
    if validate(sw):
        cands.append(sw)
    for order in [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
        sorted(customers, key=lambda v: (ready[v], dist[0, v])),
        sorted(customers, key=lambda v: dist[0, v] + dist[0, v]),
    ]:
        if time.time() > deadline:
            break
        g = greedy_construct(order)
        if validate(g):
            cands.append(g)

    for c in cands:
        d = total_distance(c)
        if d < best_d:
            best_d = d
            best_routes = c

    # Polish loop
    for _ in range(5):
        if time.time() > deadline:
            break
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt(best_routes[i])
        if time.time() > deadline:
            break
        best_routes = relocate(best_routes)
        if time.time() > deadline:
            break
        best_routes = swap(best_routes)
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt(best_routes[i])

    best_routes = [r for r in best_routes if r]
    return best_routes
# EVOLVE-BLOCK-END
