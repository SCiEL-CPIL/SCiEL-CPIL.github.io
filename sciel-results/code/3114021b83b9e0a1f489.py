"""Optimized EV routing with time windows and battery constraints.
Uses savings-based construction, battery-aware 2-opt/OR-opt, and inter-route search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

DEADLINE_SECONDS = 140.0  # safety margin below 150s limit


def _build_dist_matrix(coords):
    """Precompute Euclidean distance matrix."""
    coords = np.array(coords, dtype=np.float64)
    n = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def _simulate(route, dm, demands, ready, due, service, n_cust, stations,
              batt_max, speed, rrate, horizon, cap):
    """Simulate a route. Returns (valid, distance)."""
    # Capacity check
    load = 0.0
    for nd in route:
        if nd <= n_cust:
            load += demands[nd]
    if load > cap + 1e-9:
        return False, 0.0

    batt = batt_max
    t = 0.0
    dist = 0.0
    cur = 0

    for nd in route:
        d = dm[cur, nd]
        t += d / speed
        batt -= d
        dist += d
        if batt < -1e-9:
            return False, 0.0

        if nd <= n_cust:
            if t < ready[nd]:
                t = ready[nd]
            if t > due[nd] + 1e-9:
                return False, 0.0
            t += service[nd]
        elif nd in stations:
            need = batt_max - batt
            if need > 0:
                t += need * rrate
            batt = batt_max

        cur = nd

    # Return to depot
    d = dm[cur, 0]
    t += d / speed
    batt -= d
    dist += d
    if batt < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, dist


def _total_dist(routes, dm):
    """Total distance of all routes including depot legs."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        cur = 0
        for nd in r:
            total += dm[cur, nd]
            cur = nd
        total += dm[cur, 0]
    return total


def _route_load(route, demands, n_cust):
    return sum(demands[nd] for nd in route if nd <= n_cust)


def _savings_construct(instance, dm):
    """Clarke-Wright savings heuristic with battery/time feasibility checks."""
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_cust = instance["n_customers"]
    cap = instance["capacity"]
    batt_max = instance["battery"]
    speed = instance["speed"]
    rrate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    customers = list(range(1, n_cust + 1))

    # Compute savings: s(i,j) = d(0,i) + d(0,j) - d(i,j)
    savings = []
    for i in customers:
        for j in customers:
            if i != j:
                s = dm[0, i] + dm[0, j] - dm[i, j]
                savings.append((s, i, j))
    savings.sort(reverse=True)

    # Initialize: each customer in its own route
    routes = {c: [c] for c in customers}
    route_loads = {c: demands[c] for c in customers}
    # Track which route each customer belongs to
    cust_route = {c: c for c in customers}

    for s, i, j in savings:
        if s <= 0:
            break
        ri = cust_route[i]
        rj = cust_route[j]
        if ri == rj:
            continue
        # Check capacity
        if route_loads[ri] + route_loads[rj] > cap:
            continue
        # Check that i and j are at the ends of their routes
        route_i = routes[ri]
        route_j = routes[rj]
        if i not in (route_i[0], route_i[-1]):
            continue
        if j not in (route_j[0], route_j[-1]):
            continue

        # Try merging: we need i at the end of route_i and j at the start of route_j
        # or vice versa. Try both orientations.
        merged = None
        # Orientation 1: route_i + route_j (i at end of route_i, j at start of route_j)
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        # Orientation 2: route_j + route_i (j at end of route_j, i at start of route_i)
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        # Orientation 3: reverse route_i then + route_j
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        # Orientation 4: route_i + reverse route_j
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]

        if merged is None:
            continue

        # Check feasibility
        valid, _ = _simulate(merged, dm, demands, ready, due, service,
                             n_cust, stations, batt_max, speed, rrate, horizon, cap)
        if not valid:
            continue

        # Merge
        new_key = min(ri, rj)
        routes[new_key] = merged
        route_loads[new_key] = route_loads[ri] + route_loads[rj]
        for c in merged:
            cust_route[c] = new_key
        del routes[max(ri, rj)]
        del route_loads[max(ri, rj)]

    return [routes[k] for k in routes]


def _nearest_neighbor_construct(instance, dm):
    """Nearest neighbor construction as fallback."""
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_cust = instance["n_customers"]
    cap = instance["capacity"]
    batt_max = instance["battery"]
    speed = instance["speed"]
    rrate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    customers = list(range(1, n_cust + 1))
    customers.sort(key=lambda c: due[c])

    routes = []
    loads = []
    unvisited = set(customers)

    while unvisited:
        route = []
        load = 0.0
        cur = 0
        # Start from the customer with earliest due among unvisited
        start = min(unvisited, key=lambda c: due[c])
        route.append(start)
        load += demands[start]
        unvisited.remove(start)
        cur = start

        while unvisited:
            best = None
            best_d = float('inf')
            for c in unvisited:
                if load + demands[c] > cap:
                    continue
                d = dm[cur, c]
                if d < best_d:
                    best_d = d
                    best = c
            if best is None:
                break
            # Check feasibility
            new_route = route + [best]
            valid, _ = _simulate(new_route, dm, demands, ready, due, service,
                                 n_cust, stations, batt_max, speed, rrate, horizon, cap)
            if valid:
                route.append(best)
                load += demands[best]
                unvisited.remove(best)
                cur = best
            else:
                # Try skipping this customer
                break

        if route:
            routes.append(route)
            loads.append(load)

    return routes


def _two_opt(route, dm, demands, ready, due, service, n_cust, stations,
             batt_max, speed, rrate, horizon, cap, max_iter=100):
    """2-opt local search on a single route."""
    if len(route) < 2:
        return route

    best = route[:]
    best_d = _total_dist([best], dm)

    for _ in range(max_iter):
        improved = False
        n = len(best)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                valid, d = _simulate(new_route, dm, demands, ready, due, service,
                                     n_cust, stations, batt_max, speed, rrate, horizon, cap)
                if valid and d < best_d - 1e-9:
                    best = new_route
                    best_d = d
                    improved = True
        if not improved:
            break
    return best


def _or_opt(route, dm, demands, ready, due, service, n_cust, stations,
            batt_max, speed, rrate, horizon, cap, max_iter=50):
    """OR-opt: move 1 or 2 consecutive customers to a different position."""
    if len(route) < 3:
        return route

    best = route[:]
    best_d = _total_dist([best], dm)

    for _ in range(max_iter):
        improved = False
        n = len(best)
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                segment = best[i:i+seg_len]
                rest = best[:i] + best[i+seg_len:]
                for j in range(len(rest) + 1):
                    if i <= j <= i + seg_len - 1:
                        continue
                    new_route = rest[:j] + segment + rest[j:]
                    valid, d = _simulate(new_route, dm, demands, ready, due, service,
                                         n_cust, stations, batt_max, speed, rrate, horizon, cap)
                    if valid and d < best_d - 1e-9:
                        best = new_route
                        best_d = d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _inter_route_search(routes, dm, demands, ready, due, service, n_cust,
                        stations, batt_max, speed, rrate, horizon, cap, deadline):
    """Inter-route moves: relocate and swap customers between routes."""
    def feasible(r):
        v, _ = _simulate(r, dm, demands, ready, due, service,
                         n_cust, stations, batt_max, speed, rrate, horizon, cap)
        return v

    cur_dist = _total_dist(routes, dm)
    n_routes = len(routes)

    for _pass in range(200):
        if time.time() > deadline:
            break
        improved = False

        # Relocate moves
        for r1 in range(n_routes):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                c = routes[r1][pos1]
                if c > n_cust:
                    continue
                r1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                if not r1_new and not feasible([]):
                    continue
                if r1_new and not feasible(r1_new):
                    continue

                for r2 in range(n_routes):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if _route_load(routes[r2], demands, n_cust) + demands[c] > cap:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        r2_new = routes[r2][:pos2] + [c] + routes[r2][pos2:]
                        if feasible(r2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = r1_new
                            new_routes[r2] = r2_new
                            nd = _total_dist(new_routes, dm)
                            if nd < cur_dist - 1e-9:
                                routes = new_routes
                                cur_dist = nd
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Swap moves
        if not improved and time.time() < deadline:
            for r1 in range(n_routes):
                if time.time() > deadline:
                    break
                for pos1 in range(len(routes[r1])):
                    if time.time() > deadline:
                        break
                    c1 = routes[r1][pos1]
                    if c1 > n_cust:
                        continue
                    for r2 in range(n_routes):
                        if time.time() > deadline:
                            break
                        if r1 == r2:
                            continue
                        for pos2 in range(len(routes[r2])):
                            if time.time() > deadline:
                                break
                            c2 = routes[r2][pos2]
                            if c2 > n_cust:
                                continue
                            # Check capacity
                            l1 = _route_load(routes[r1], demands, n_cust)
                            l2 = _route_load(routes[r2], demands, n_cust)
                            if l1 - demands[c1] + demands[c2] > cap:
                                continue
                            if l2 - demands[c2] + demands[c1] > cap:
                                continue
                            r1_new = routes[r1][:]
                            r1_new[pos1] = c2
                            r2_new = routes[r2][:]
                            r2_new[pos2] = c1
                            if feasible(r1_new) and feasible(r2_new):
                                new_routes = [r[:] for r in routes]
                                new_routes[r1] = r1_new
                                new_routes[r2] = r2_new
                                nd = _total_dist(new_routes, dm)
                                if nd < cur_dist - 1e-9:
                                    routes = new_routes
                                    cur_dist = nd
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

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _validate(routes, instance):
    """Full validation."""
    dm = _build_dist_matrix(instance["coords"])
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_cust = instance["n_customers"]
    cap = instance["capacity"]
    batt_max = instance["battery"]
    speed = instance["speed"]
    rrate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    served = set()
    total = 0.0
    for r in routes:
        v, d = _simulate(r, dm, demands, ready, due, service,
                         n_cust, stations, batt_max, speed, rrate, horizon, cap)
        if not v:
            return False, 0.0
        total += d
        for nd in r:
            if nd <= n_cust:
                if nd in served:
                    return False, 0.0
                served.add(nd)

    if served != set(range(1, n_cust + 1)):
        return False, 0.0
    return True, total


def plan(instance):
    """Main entry point."""
    start = time.time()
    deadline = start + DEADLINE_SECONDS

    dm = _build_dist_matrix(instance["coords"])
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_cust = instance["n_customers"]
    cap = instance["capacity"]
    batt_max = instance["battery"]
    speed = instance["speed"]
    rrate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    # Try multiple constructions
    candidates = []

    # Savings-based
    if time.time() < deadline - 20:
        routes_s = _savings_construct(instance, dm)
        v, d = _validate(routes_s, instance)
        if v:
            candidates.append((d, routes_s))

    # Nearest neighbor
    if time.time() < deadline - 20:
        routes_nn = _nearest_neighbor_construct(instance, dm)
        v, d = _validate(routes_nn, instance)
        if v:
            candidates.append((d, routes_nn))

    # Fallback: one per route
    if not candidates:
        routes_fb = [[c] for c in range(1, n_cust + 1)]
        v, d = _validate(routes_fb, instance)
        if v:
            candidates.append((d, routes_fb))
        else:
            return [[c] for c in range(1, n_cust + 1)]

    candidates.sort(key=lambda x: x[0])
    best_d, best_routes = candidates[0]

    # Local search
    if time.time() < deadline - 5:
        routes = best_routes[:]

        # Intra-route 2-opt and OR-opt
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            routes[i] = _two_opt(routes[i], dm, demands, ready, due, service,
                                 n_cust, stations, batt_max, speed, rrate, horizon, cap,
                                 max_iter=80)
            routes[i] = _or_opt(routes[i], dm, demands, ready, due, service,
                                n_cust, stations, batt_max, speed, rrate, horizon, cap,
                                max_iter=30)

        # Inter-route search
        if time.time() < deadline - 3:
            routes = _inter_route_search(routes, dm, demands, ready, due, service,
                                         n_cust, stations, batt_max, speed, rrate,
                                         horizon, cap, deadline - 2)

        # Re-validate
        v, d = _validate(routes, instance)
        if v and d < best_d:
            best_routes = routes
            best_d = d

    # Final safety check
    v, d = _validate(best_routes, instance)
    if not v:
        # Fallback
        routes_fb = [[c] for c in range(1, n_cust + 1)]
        v2, _ = _validate(routes_fb, instance)
        if v2:
            return routes_fb
        return best_routes

    return best_routes

# EVOLVE-BLOCK-END
