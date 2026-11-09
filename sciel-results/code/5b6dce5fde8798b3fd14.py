"""EV routing with time windows + battery: regret-2 construction + fast 2-opt/Or-opt + station moves."""
import math
import time

import numpy as np

_DEADLINE = 118.0  # wall-clock safety margin (evaluator kills at 150s)
_SPEED = 1.0


def _dist_matrix(instance):
    coords = np.asarray(instance["coords"], dtype=float)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    return d


def _check_route(route, d, inst):
    """Full feasibility + (dist, time). Returns None if infeasible, else (dist, time)."""
    cap = inst["capacity"]
    bat_cap = inst["battery"]
    rc = inst["recharge_rate"]
    horizon = inst["horizon"]
    demand = inst["demand"]
    ready = inst["ready"]
    due = inst["due"]
    service = inst["service"]
    is_station = inst["_is_station"]
    n = len(route)
    load = 0
    bat = bat_cap
    t = 0.0
    dist = 0.0
    prev = 0
    for i in range(n):
        node = route[i]
        dd = d[prev, node]
        if dd > bat:
            return None
        t += dd / _SPEED
        bat -= dd
        dist += dd
        if is_station[node]:
            t += (bat - 0.0) * rc
            bat = bat_cap
        else:
            if t > due[node]:
                return None
            t = max(t, ready[node]) + service[node]
            load += demand[node]
            if load > cap:
                return None
        prev = node
    dd = d[prev, 0]
    if dd > bat:
        return None
    t += dd / _SPEED
    if t > horizon:
        return None
    return dist, t


def _construct(instance):
    n_cust = instance["n_customers"]
    coords = np.asarray(instance["coords"], dtype=float)
    demand = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    is_station = np.zeros(len(coords), dtype=bool)
    for s in stations:
        is_station[s] = True
    d = _dist_matrix(instance)
    cap = instance["capacity"]
    bat_cap = instance["battery"]
    rc = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute quick feasibility of a single customer from depot (with/without a station).
    # We keep it simple: build routes greedily with a full simulation check per insertion.
    customers = list(range(1, n_cust + 1))
    customers.sort(key=lambda v: due[v])
    remaining = set(customers)
    routes = []  # each: list of node ids

    def route_cost(route):
        r = _check_route(route, d, instance)
        return r[0] if r is not None else float("inf")

    def try_insert(route, node, allow_station):
        # find best position (and optional single station before node) giving feasible min cost
        best = None
        L = len(route)
        candidates = list(range(L + 1))
        # try inserting customer at each position
        for pos in candidates:
            cand = route[:pos] + [node] + route[pos:]
            r = _check_route(cand, d, instance)
            if r is None:
                continue
            cost = r[0]
            if best is None or cost < best[0]:
                best = (cost, cand)
        if best is not None:
            return best[1], best[0]
        # try adding a station just before the customer at each position
        if allow_station:
            for s in stations:
                for pos in candidates:
                    cand = route[:pos] + [s, node] + route[pos:]
                    r = _check_route(cand, d, instance)
                    if r is None:
                        continue
                    cost = r[0]
                    if best is None or cost < best[0]:
                        best = (cost, cand)
        if best is not None:
            return best[1], best[0]
        return None, float("inf")

    # Regret-2 greedy
    while remaining:
        # compute regret for each remaining customer over current routes + new route
        regrets = {}
        for node in remaining:
            costs = []
            for route in routes:
                _, c = try_insert(route, node, allow_station=False)
                if c < float("inf"):
                    costs.append(c)
            # new route cost
            r = _check_route([node], d, instance)
            if r is not None:
                costs.append(r[0])
            if not costs:
                # must use a station somewhere; try new route with station
                ok = False
                for s in stations:
                    rr = _check_route([s, node], d, instance)
                    if rr is not None:
                        costs.append(rr[0])
                        ok = True
                        break
                if not ok:
                    # fallback: single customer route (should be feasible per contract)
                    costs.append(0.0)
            costs.sort()
            regret = (costs[1] - costs[0]) if len(costs) >= 2 else 0.0
            regrets[node] = (regret, -due[node])
        # pick max regret, tie-break by earliest due
        node = max(remaining, key=lambda v: (regrets[v][0], regrets[v][1]))
        # decide where to insert (min cost), allowing station if needed
        best_cost = float("inf")
        best_choice = None  # ("new", route) or ("route", idx, route)
        for idx, route in enumerate(routes):
            newroute, c = try_insert(route, node, allow_station=False)
            if c < best_cost:
                best_cost = c
                best_choice = ("route", idx, newroute)
        r = _check_route([node], d, instance)
        if r is not None and r[0] < best_cost:
            best_cost = r[0]
            best_choice = ("new", [node])
        if best_choice is None:
            # need a station
            for s in stations:
                rr = _check_route([s, node], d, instance)
                if rr is not None:
                    best_choice = ("new", [s, node])
                    break
        if best_choice is None:
            best_choice = ("new", [node])
        if best_choice[0] == "route":
            routes[best_choice[1]] = best_choice[2]
        else:
            routes.append(best_choice[1])
        remaining.discard(node)

    return routes, d


def _local_search(routes, d, instance, deadline):
    cap = instance["capacity"]
    stations = set(instance["stations"])
    is_station = instance["_is_station"]
    n_routes = len(routes)

    def valid(route):
        return _check_route(route, d, instance) is not None

    def cost(route):
        r = _check_route(route, d, instance)
        return r[0] if r is not None else float("inf")

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for ri in range(n_routes):
            if time.time() >= deadline:
                return routes
            route = routes[ri]
            L = len(route)
            if L < 2:
                continue
            # 2-opt
            for i in range(L - 1):
                if time.time() >= deadline:
                    return routes
                for j in range(i + 1, L):
                    newroute = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if valid(newroute) and cost(newroute) < cost(route) - 1e-9:
                        route = newroute
                        routes[ri] = route
                        improved = True
                        L = len(route)
            # Or-opt: move 1 or 2 consecutive nodes to another position
            for seg in (1, 2):
                for i in range(L - seg + 1):
                    if time.time() >= deadline:
                        return routes
                    block = route[i:i + seg]
                    base = route[:i] + route[i + seg:]
                    for pos in range(len(base) + 1):
                        if pos <= i or pos >= i + seg:
                            continue
                        newroute = base[:pos] + block + base[pos:]
                        if valid(newroute) and cost(newroute) < cost(route) - 1e-9:
                            route = newroute
                            routes[ri] = route
                            improved = True
                            L = len(route)
                            break
            # Station insertion: try inserting a station after each customer to help battery
            L = len(route)
            for pos in range(L + 1):
                if time.time() >= deadline:
                    return routes
                for s in stations:
                    if is_station[s]:
                        continue
                    newroute = route[:pos] + [s] + route[pos:]
                    if valid(newroute) and cost(newroute) < cost(route) - 1e-9:
                        route = newroute
                        routes[ri] = route
                        improved = True
                        L = len(route)
                        break
    return routes


def plan(instance):
    t0 = time.time()
    coords = np.asarray(instance["coords"], dtype=float)
    is_station = np.zeros(len(coords), dtype=bool)
    for s in instance["stations"]:
        is_station[s] = True
    instance = dict(instance)
    instance["_is_station"] = is_station

    routes, d = _construct(instance)

    # Ensure every customer exactly once; if construction left any out (shouldn't), add them.
    served = set()
    for r in routes:
        for n in r:
            if not is_station[n]:
                served.add(n)
    missing = set(range(1, instance["n_customers"] + 1)) - served
    for m in missing:
        routes.append([m])

    # Local search with deadline guard
    deadline = t0 + (_DEADLINE - (time.time() - t0)) * 0.9
    routes = _local_search(routes, d, instance, deadline)

    # Final safety: drop nothing; verify feasibility, if a route is infeasible fall back to singles
    final = []
    for r in routes:
        if _check_route(r, d, instance) is None:
            # split into individual customers to guarantee feasibility
            for n in r:
                if not is_station[n]:
                    final.append([n])
        else:
            final.append(r)
    return final
