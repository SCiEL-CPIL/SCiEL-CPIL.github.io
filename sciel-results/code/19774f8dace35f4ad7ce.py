"""EV routing with time windows and battery/charging. Regret-insertion + tabu local search."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from scipy.spatial import cKDTree

_TIME_BUDGET = 110.0


def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def _simulate(route, coords, demand, ready, due, service, stations,
              capacity, battery_max, speed, recharge_rate, horizon):
    t = 0.0
    bat = float(battery_max)
    load = 0.0
    total = 0.0
    prev = 0
    for n in route:
        d = _dist(coords[prev], coords[n])
        t += d / speed
        bat -= d
        total += d
        if bat < -1e-7:
            return False, total
        if n in stations:
            if bat < 0.0:
                return False, total
            t += max(0.0, bat) * recharge_rate
            bat = float(battery_max)
        else:
            if t > due[n] + 1e-7:
                return False, total
            t = max(t, ready[n])
            t += service[n]
            load += demand[n]
            if load > capacity + 1e-7:
                return False, total
        prev = n
    d = _dist(coords[prev], coords[0])
    t += d / speed
    bat -= d
    total += d
    if bat < -1e-7:
        return False, total
    if t > horizon + 1e-7:
        return False, total
    return True, total


def _insert(route, idx, node, coords, demand, ready, due, service, stations,
            capacity, battery_max, speed, recharge_rate, horizon):
    cand = route[:idx] + [node] + route[idx:]
    return _simulate(cand, coords, demand, ready, due, service, stations,
                     capacity, battery_max, speed, recharge_rate, horizon)


def _construct(instance, rng, time_limit):
    coords = instance["coords"]
    demand = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_cust = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    ids = list(range(1, n_cust + 1))
    routes = []
    route_load = []
    route_dist = []
    remaining = set(ids)

    def try_build_route():
        if not remaining:
            return None
        seed = rng.choice(sorted(remaining))
        ok, total = _simulate([seed], coords, demand, ready, due, service, stations,
                              capacity, battery_max, speed, recharge_rate, horizon)
        if not ok:
            return None
        route = [seed]
        load = demand[seed]
        total = total
        while True:
            best = None
            best_cost = None
            for n in sorted(remaining):
                if load + demand[n] > capacity + 1e-7:
                    continue
                for idx in range(len(route) + 1):
                    ok, tot = _insert(route, idx, n, coords, demand, ready, due,
                                      service, stations, capacity, battery_max,
                                      speed, recharge_rate, horizon)
                    if ok:
                        cost = tot - total
                        if best_cost is None or cost < best_cost:
                            best_cost = cost
                            best = (idx, n, tot)
            if best is None:
                break
            idx, n, tot = best
            route = route[:idx] + [n] + route[idx:]
            load += demand[n]
            total = tot
            remaining.discard(n)
        return route, load, total

    deadline = time.time() + time_limit
    for _ in range(4):
        if time.time() > deadline:
            break
        res = try_build_route()
        if res is None:
            break
        route, load, total = res
        routes.append(route)
        route_load.append(load)
        route_dist.append(total)

    while remaining:
        if time.time() > deadline:
            break
        best_node = None
        best_regret = -1.0
        for n in sorted(remaining):
            costs = []
            for ri, route in enumerate(routes):
                if route_load[ri] + demand[n] > capacity + 1e-7:
                    continue
                best_here = None
                for idx in range(len(route) + 1):
                    ok, tot = _insert(route, idx, n, coords, demand, ready, due,
                                      service, stations, capacity, battery_max,
                                      speed, recharge_rate, horizon)
                    if ok:
                        cost = tot - route_dist[ri]
                        if best_here is None or cost < best_here:
                            best_here = cost
                if best_here is not None:
                    costs.append(best_here)
            if len(costs) < 2:
                regret = (costs[0] if costs else float('inf')) + 1000.0
            else:
                costs.sort()
                regret = costs[1] - costs[0]
            if best_node is None or regret > best_regret:
                best_regret = regret
                best_node = n
        if best_node is None:
            break
        n = best_node
        best_ri = None
        best_cost = None
        best_idx = None
        best_tot = None
        for ri, route in enumerate(routes):
            if route_load[ri] + demand[n] > capacity + 1e-7:
                continue
            for idx in range(len(route) + 1):
                ok, tot = _insert(route, idx, n, coords, demand, ready, due,
                                  service, stations, capacity, battery_max,
                                  speed, recharge_rate, horizon)
                if ok:
                    cost = tot - route_dist[ri]
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_ri = ri
                        best_idx = idx
                        best_tot = tot
        if best_ri is None:
            ok, tot = _simulate([n], coords, demand, ready, due, service, stations,
                                capacity, battery_max, speed, recharge_rate, horizon)
            if not ok:
                return None
            routes.append([n])
            route_load.append(demand[n])
            route_dist.append(tot)
        else:
            route = routes[best_ri][:best_idx] + [n] + routes[best_ri][best_idx:]
            routes[best_ri] = route
            route_load[best_ri] += demand[n]
            route_dist[best_ri] = best_tot
        remaining.discard(n)

    if not remaining and len(routes) > 0:
        return routes
    return None


def _total_distance(routes, coords):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for n in route:
            total += _dist(coords[prev], coords[n])
            prev = n
        total += _dist(coords[prev], coords[0])
    return total


def _tabu_search(routes, instance, rng, time_limit):
    coords = instance["coords"]
    demand = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    def simulate(route):
        return _simulate(route, coords, demand, ready, due, service, stations,
                         capacity, battery_max, speed, recharge_rate, horizon)

    current = [list(r) for r in routes]
    route_cost = []
    cur_cost = 0.0
    for r in current:
        ok, tot = simulate(r)
        if not ok:
            return None
        route_cost.append(tot)
        cur_cost += tot
    best = [list(r) for r in current]
    best_cost = cur_cost

    deadline = time.time() + time_limit
    tabu_len = 10
    tabu = {}
    it = 0
    no_improve = 0
    while time.time() < deadline:
        it += 1
        moves = []
        # 2-opt within a route
        for ri, route in enumerate(current):
            L = len(route)
            for i in range(L - 1):
                for j in range(i + 1, L):
                    new = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    ok, tot = simulate(new)
                    if ok:
                        delta = tot - route_cost[ri]
                        moves.append((delta, "2opt", ri, i, j, new))
        # move node between routes
        for ri, route in enumerate(current):
            for pos, n in enumerate(route):
                for rj in range(len(current)):
                    if rj == ri:
                        continue
                    if demand[n] + sum(demand[x] for x in current[rj]) > capacity + 1e-7:
                        continue
                    for idx in range(len(current[rj]) + 1):
                        cand = list(current[rj])
                        cand.insert(idx, n)
                        ok, tot = simulate(cand)
                        if not ok:
                            continue
                        ok2, tot2 = simulate(route[:pos] + route[pos + 1:])
                        if not ok2:
                            continue
                        delta = (tot + tot2) - (route_cost[ri] + route_cost[rj])
                        moves.append((delta, "move", ri, pos, rj, idx, n))
        if not moves:
            break
        moves.sort(key=lambda m: m[0])
        improved = False
        for mv in moves:
            kind = mv[1]
            if kind == "2opt":
                _, _, ri, i, j, new = mv
                key = (ri, "2opt", i, j)
                if tabu.get(key, -1) > it:
                    continue
                new_rc = list(route_cost)
                new_rc[ri] = None
                okc = True
                cc = 0.0
                trial = [list(r) for r in current]
                trial[ri] = new
                for r in trial:
                    ok, tot = simulate(r)
                    if not ok:
                        okc = False
                        break
                    cc += tot
                if not okc:
                    continue
                current = trial
                route_cost = []
                for r in current:
                    _, tot = simulate(r)
                    route_cost.append(tot)
                cur_cost = cc
                tabu[key] = it + tabu_len
                if cur_cost < best_cost - 1e-9:
                    best_cost = cur_cost
                    best = [list(r) for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                break
            elif kind == "move":
                _, _, ri, pos, rj, idx, n = mv
                key = (ri, pos, rj, idx)
                if tabu.get(key, -1) > it:
                    continue
                trial = [list(r) for r in current]
                trial[ri].pop(pos)
                trial[rj].insert(idx, n)
                okc = True
                cc = 0.0
                for r in trial:
                    ok, tot = simulate(r)
                    if not ok:
                        okc = False
                        break
                    cc += tot
                if not okc:
                    continue
                current = trial
                route_cost = []
                for r in current:
                    _, tot = simulate(r)
                    route_cost.append(tot)
                cur_cost = cc
                tabu[key] = it + tabu_len
                if cur_cost < best_cost - 1e-9:
                    best_cost = cur_cost
                    best = [list(r) for r in current]
                    no_improve = 0
                else:
                    no_improve += 1
                improved = True
                break
        if not improved:
            break
        if it % tabu_len == 0:
            tabu = {}
        if no_improve > 80:
            break
    ok = True
    for r in best:
        okk, _ = simulate(r)
        if not okk:
            ok = False
    if not ok:
        return None
    return best


def plan(instance):
    t0 = time.time()
    rng = random.Random(12345)
    routes = _construct(instance, rng, 12.0)
    if routes is None:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
    elapsed = time.time() - t0
    remaining = max(1.0, _TIME_BUDGET - elapsed - 5.0)
    improved = _tabu_search(routes, instance, rng, remaining)
    if improved is not None:
        routes = improved
    coords = instance["coords"]
    ok = True
    for r in routes:
        okk, _ = _simulate(r, coords, instance["demand"], instance["ready"],
                           instance["due"], instance["service"], set(instance["stations"]),
                           instance["capacity"], instance["battery"], instance["speed"],
                           instance["recharge_rate"], instance["horizon"])
        if not okk:
            ok = False
            break
    if not ok:
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        return [[v] for v in customers]
    return routes
# EVOLVE-BLOCK-END
