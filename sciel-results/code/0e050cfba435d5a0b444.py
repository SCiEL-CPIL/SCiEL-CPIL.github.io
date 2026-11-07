"""EV routing with time windows: robust construction + local search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def plan(instance):
    t0 = time.time()
    DEADLINE = 138.0

    C = instance["n_customers"]
    coords = instance["coords"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        xi, yi = coords[i]
        for j in range(i + 1, n_nodes):
            xj, yj = coords[j]
            dd = math.hypot(xi - xj, yi - yj)
            D[i, j] = dd
            D[j, i] = dd

    customers = list(range(1, C + 1))

    def simulate(route):
        cur = 0
        t = 0.0
        bat = float(battery)
        rem = float(capacity)
        dist = 0.0
        for c in route:
            d = D[cur, c]
            if d > bat + 1e-9:
                return dist, False
            bat -= d
            t += d / speed
            dist += d
            if t > due[c] + 1e-9:
                return dist, False
            t = max(t, ready[c])
            if t > due[c] + 1e-9:
                return dist, False
            t += service[c]
            rem -= demands[c]
            if rem < -1e-9:
                return dist, False
            cur = c
        d = D[cur, 0]
        if d > bat + 1e-9:
            return dist, False
        bat -= d
        t += d / speed
        dist += d
        if t > horizon + 1e-9:
            return dist, False
        return dist, True

    def arrival(cur, t, bat, c):
        """Return arrival time at customer c from cur with battery bat, or None."""
        d = D[cur, c]
        if d <= bat + 1e-9:
            a = t + d / speed
            a = max(a, ready[c])
            return a if a <= due[c] + 1e-9 else None
        for s in stations:
            d1 = D[cur, s]
            if d1 > bat + 1e-9:
                continue
            bat1 = bat - d1
            t1 = t + d1 / speed
            t2 = t1 + bat1 * recharge_rate
            d2 = D[s, c]
            a = t2 + d2 / speed
            a = max(a, ready[c])
            if a <= due[c] + 1e-9:
                return a
        return None

    def battery_after(route):
        cur = 0
        bat = float(battery)
        for n in route:
            d = D[cur, n]
            if d > bat + 1e-9:
                return None
            bat -= d
            if n in stations:
                bat = float(battery)
            cur = n
        return bat

    # Construction: nearest feasible customer
    unvisited = set(customers)
    routes = []
    while unvisited:
        route = []
        rem = float(capacity)
        cur = 0
        t = 0.0
        bat = float(battery)
        while unvisited:
            best = None
            best_arr = None
            for c in unvisited:
                if demands[c] > rem + 1e-9:
                    continue
                a = arrival(cur, t, bat, c)
                if a is None:
                    continue
                if best is None or a < best_arr:
                    best = c
                    best_arr = a
            if best is None:
                break
            route.append(best)
            unvisited.discard(best)
            rem -= demands[best]
            cur = best
            t = best_arr + service[best]
            bat = battery_after(route)
            if bat is None:
                unvisited.add(best)
                route.pop()
                break
        if route:
            routes.append(route)

    # Local search
    def rdist(route):
        d, ok = simulate(route)
        return d if ok else float('inf')

    # 2-opt within routes
    for _ in range(3):
        if time.time() - t0 > DEADLINE:
            break
        improved = False
        for ri in range(len(routes)):
            route = routes[ri]
            n = len(route)
            base = rdist(route)
            for i in range(n - 1):
                if time.time() - t0 > DEADLINE:
                    break
                for j in range(i + 1, n):
                    newroute = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    nd = rdist(newroute)
                    if nd < base - 1e-9:
                        routes[ri] = newroute
                        base = nd
                        improved = True
                        break
        if not improved:
            break

    # Relocate between routes
    for _ in range(3):
        if time.time() - t0 > DEADLINE:
            break
        improved = False
        for ri in range(len(routes)):
            if time.time() - t0 > DEADLINE:
                break
            route = routes[ri]
            for ci in range(len(route)):
                cust = route[ci]
                rest = route[:ci] + route[ci + 1:]
                rest_d = rdist(rest)
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    target = routes[rj]
                    base = rdist(target)
                    best_ins = None
                    best_nd = float('inf')
                    for pos in range(len(target) + 1):
                        cand = target[:pos] + [cust] + target[pos:]
                        nd = rdist(cand)
                        if nd < best_nd:
                            best_nd = nd
                            best_ins = cand
                    if best_ins is not None and (best_nd + rest_d) < (rdist(route) + base) - 1e-9:
                        routes[ri] = rest
                        routes[rj] = best_ins
                        improved = True
                        break
                if improved:
                    break
        routes = [r for r in routes if r]
        if not improved:
            break

    # Validity guard
    for r in routes:
        d, ok = simulate(r)
        if not ok:
            return [[c] for c in customers]
    return routes
# EVOLVE-BLOCK-END
