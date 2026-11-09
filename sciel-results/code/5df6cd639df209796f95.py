"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_time_and_battery(route, D, instance):
    """Return (total_time, total_dist, min_battery, feasible) for a route."""
    speed = instance.get("speed", 1)
    battery_cap = instance.get("battery", 300)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    t = 0.0
    bat = battery_cap
    dist = 0.0
    prev = 0
    min_bat = battery_cap
    for v in route:
        d = D[prev, v]
        t += d / speed
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, dist, bat, False
        min_bat = min(min_bat, bat)
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = battery_cap
        else:
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, dist, bat, False
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d / speed
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, dist, bat, False
    min_bat = min(min_bat, bat)
    if t > horizon + 1e-9:
        return t, dist, bat, False
    return t, dist, min_bat, True


def _route_feasible(route, D, instance):
    if not route:
        return True
    cap = instance.get("capacity", 100)
    demands = instance["demands"]
    if sum(demands[v] for v in route) > cap + 1e-9:
        return False
    _, _, _, feas = _route_time_and_battery(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _construct_greedy(instance, D, deadline):
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: instance["due"][v])

    routes = []
    unassigned = set(customers)
    battery_cap = instance.get("battery", 300)
    speed = instance.get("speed", 1)
    recharge_rate = instance.get("recharge_rate", 0.5)
    cap = instance.get("capacity", 100)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    while unassigned:
        if time.time() > deadline:
            break
        route = []
        prev = 0
        t = 0.0
        bat = battery_cap
        demand = 0

        while unassigned:
            if time.time() > deadline:
                break
            best_v = None
            best_cost = float('inf')
            for v in unassigned:
                d = D[prev, v]
                new_t = t + d / speed
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                if demand + demands[v] > cap + 1e-9:
                    continue
                if v in stations:
                    new_t += (battery_cap - new_bat) * recharge_rate
                    new_bat = battery_cap
                else:
                    new_t = max(new_t, ready[v])
                    if new_t > due[v] + 1e-9:
                        continue
                    new_t += service[v]
                ret_d = D[v, 0]
                ret_t = new_t + ret_d / speed
                ret_bat = new_bat - ret_d
                if ret_t > horizon + 1e-9 or ret_bat < -1e-9:
                    continue
                cost = d + 0.01 * due[v]
                if cost < best_cost:
                    best_cost = cost
                    best_v = v
            if best_v is None:
                break
            d = D[prev, best_v]
            t += d / speed
            bat -= d
            demand += demands[best_v]
            if best_v in stations:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            else:
                t = max(t, ready[best_v])
                t += service[best_v]
            route.append(best_v)
            prev = best_v
            unassigned.discard(best_v)

        if route:
            routes.append(route)
        else:
            v = min(unassigned, key=lambda x: due[x])
            routes.append([v])
            unassigned.discard(v)

    return routes


def _2opt(route, D, instance, max_iter=200):
    if len(route) < 2:
        return route
    best = route[:]

    for _ in range(max_iter):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt(route, D, instance, max_iter=100):
    if len(route) < 3:
        return route
    best = route[:]

    for _ in range(max_iter):
        improved = False
        for i in range(len(best)):
            node = best[i]
            for j in range(len(best)):
                if j == i or j == i - 1:
                    continue
                new_route = best[:i] + best[i + 1:]
                ins_j = j if i > j else j - 1
                if ins_j < 0:
                    ins_j = 0
                if ins_j > len(new_route):
                    ins_j = len(new_route)
                candidate = new_route[:ins_j] + [node] + new_route[ins_j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                if _route_dist(candidate, D) < _route_dist(best, D) - 1e-9:
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _inter_route_2opt(routes, D, instance, max_iters=30):
    for _ in range(max_iters):
        improved = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140

    coords = instance["coords"]
    D = _dist_matrix(coords)

    routes = _construct_greedy(instance, D, deadline)

    if time.time() > deadline:
        return routes

    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, instance)
        routes[r_idx] = _or_opt(routes[r_idx], D, instance)

    if time.time() > deadline:
        return routes

    routes = _inter_route_2opt(routes, D, instance)

    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, instance)

    return routes
# EVOLVE-BLOCK-END
