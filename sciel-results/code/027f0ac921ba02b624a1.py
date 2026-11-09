"""EV routing with time windows and battery; construction + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    N = len(coords)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    cust = list(range(1, n_customers + 1))

    def check_route(route):
        # route: list of customer ids (depot implicit at both ends)
        if not route:
            return True, 0.0, 0.0
        total_demand = demands[route].sum()
        if total_demand > capacity:
            return False, 0.0, 0.0
        t = 0.0
        bat = float(battery)
        d = 0.0
        prev = 0
        for v in route:
            dd = dist[prev, v]
            if bat < dd - 1e-9:
                return False, 0.0, 0.0
            bat -= dd
            t += dd / speed
            d += dd
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        # return to depot
        dd = dist[prev, 0]
        if bat < dd - 1e-9:
            return False, 0.0, 0.0
        bat -= dd
        t += dd / speed
        d += dd
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, d, t

    def try_insert(route, v, pos):
        newroute = route[:pos] + [v] + route[pos:]
        ok, d, t = check_route(newroute)
        if not ok:
            return None
        return newroute, d

    def route_distance(route):
        if not route:
            return 0.0
        d = 0.0
        prev = 0
        for v in route:
            d += dist[prev, v]
            prev = v
        d += dist[prev, 0]
        return d

    # ---------- Construction ----------
    routes = []
    remaining = list(cust)
    while remaining:
        route = []
        while True:
            if time.time() > deadline:
                break
            best = None
            best_key = None
            for v in remaining:
                for pos in range(len(route) + 1):
                    res = try_insert(route, v, pos)
                    if res is None:
                        continue
                    newroute, d = res
                    key = (d,)
                    if best_key is None or key < best_key:
                        best_key = key
                        best = (v, newroute)
            if best is None:
                break
            v, newroute = best
            route = newroute
            remaining.remove(v)
        if route:
            routes.append(route)

    # ---------- Local search ----------
    def total_dist():
        s = 0.0
        for r in routes:
            s += route_distance(r)
        return s

    best_total = total_dist()
    best_routes = [list(r) for r in routes]

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Relocate
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for j in range(len(routes[i])):
                v = routes[i][j]
                src = routes[i]
                src_without = src[:j] + src[j + 1:]
                best_gain = 1e-9
                best_target = None
                best_pos = None
                for k in range(len(routes)):
                    if k == i:
                        continue
                    tgt = routes[k]
                    for pos in range(len(tgt) + 1):
                        new_tgt = tgt[:pos] + [v] + tgt[pos:]
                        ok_t, _, _ = check_route(new_tgt)
                        if not ok_t:
                            continue
                        ok_s, _, _ = check_route(src_without)
                        if not ok_s:
                            continue
                        d_t = route_distance(new_tgt)
                        d_s = route_distance(src_without)
                        d0_t = route_distance(tgt)
                        d0_s = route_distance(src)
                        gain = (d_t + d_s) - (d0_t + d0_s)
                        if -gain > best_gain:
                            best_gain = -gain
                            best_target = k
                            best_pos = pos
                            best_src_without = src_without
                            best_new_tgt = new_tgt
                if best_target is not None:
                    routes[i] = best_src_without
                    routes[best_target] = best_new_tgt
                    if not routes[i]:
                        routes.pop(i)
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue

        # Swap
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            for k in range(i + 1, len(routes)):
                ri = routes[i]
                rk = routes[k]
                for a in range(len(ri)):
                    for b in range(len(rk)):
                        vi = ri[a]
                        vk = rk[b]
                        new_ri = list(ri)
                        new_rk = list(rk)
                        new_ri[a] = vk
                        new_rk[b] = vi
                        ok_i, _, _ = check_route(new_ri)
                        ok_k, _, _ = check_route(new_rk)
                        if not (ok_i and ok_k):
                            continue
                        d_i = route_distance(new_ri)
                        d_k = route_distance(new_rk)
                        d0_i = route_distance(ri)
                        d0_k = route_distance(rk)
                        if d_i + d_k < d0_i + d0_k - 1e-9:
                            routes[i] = new_ri
                            routes[k] = new_rk
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
        if improved:
            continue

        # 2-opt intra route
        for i in range(len(routes)):
            if time.time() > deadline:
                break
            ri = routes[i]
            n = len(ri)
            if n < 2:
                continue
            found = False
            for a in range(n - 1):
                for b in range(a + 1, n):
                    new_ri = ri[:a] + ri[a:b + 1][::-1] + ri[b + 1:]
                    ok, _, _ = check_route(new_ri)
                    if not ok:
                        continue
                    d = route_distance(new_ri)
                    d0 = route_distance(ri)
                    if d < d0 - 1e-9:
                        routes[i] = new_ri
                        ri = new_ri
                        n = len(ri)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if improved:
            continue

    cur = total_dist()
    if cur < best_total:
        best_total = cur
        best_routes = [list(r) for r in routes]

    return best_routes
# EVOLVE-BLOCK-END
