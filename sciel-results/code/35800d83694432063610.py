# EVOLVE-BLOCK-START
import math
import time
import numpy as np


def plan(instance):
    t0 = time.time()
    DEADLINE = 140.0          # hard wall-clock guard (evaluator kills at 150s)
    MARGIN = 6.0              # stop searching this far before the limit

    coords = np.asarray(instance["coords"], dtype=float)
    demand = np.asarray(instance["demand"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_cust = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])
    N = coords.shape[0]

    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))

    def dist(a, b):
        return D[a, b]

    def feasible(route):
        """Return (ok, total_distance). route = list of customer ids."""
        t = 0.0
        b = battery
        load = 0.0
        prev = 0
        total = 0.0
        for c in route:
            d = dist(prev, c)
            if b - d < -1e-6:
                return False, 0.0
            b -= d
            t += d / speed
            if t > due[c] + 1e-6:
                return False, 0.0
            t = max(t, ready[c])
            if t > due[c] + 1e-6:
                return False, 0.0
            t += service[c]
            load += demand[c]
            if load > capacity + 1e-6:
                return False, 0.0
            prev = c
        d = dist(prev, 0)
        if b - d < -1e-6:
            return False, 0.0
        t += d / speed
        if t > horizon + 1e-6:
            return False, 0.0
        total += d
        return True, total

    def total_distance(routes):
        s = 0.0
        for r in routes:
            if not r:
                continue
            prev = 0
            for c in r:
                s += dist(prev, c)
                prev = c
            s += dist(prev, 0)
        return s

    customers = list(range(1, n_cust + 1))
    # earliest-due first construction order
    order = sorted(customers, key=lambda v: due[v])

    # ---- Construction: Clarke-Wright savings ----
    # savings[i] = dist(0,i)+dist(0,j)-dist(i,j) for merging routes ending i, starting j
    # We build routes by repeatedly taking the best feasible merge.
    # Represent each route as a list; track endpoints.
    routes = [[c] for c in order]

    def try_merge(ri, rj):
        """Try to merge route rj into route ri in the best of 4 orientations."""
        a = routes[ri]
        b = routes[rj]
        if not a or not b:
            return None
        candidates = [a + b, list(reversed(a)) + b, a + list(reversed(b)),
                      list(reversed(a)) + list(reversed(b))]
        best = None
        best_d = None
        for cand in candidates:
            ok, d = feasible(cand)
            if ok:
                if best_d is None or d < best_d:
                    best_d = d
                    best = cand
        return best

    # Precompute savings to guide merge order
    savings = []
    for i in range(n_cust):
        for j in range(i + 1, n_cust):
            s = dist(0, i + 1) + dist(0, j + 1) - dist(i + 1, j + 1)
            savings.append((s, i + 1, j + 1))
    savings.sort(reverse=True)

    # map customer -> route index
    cust_route = {c: i for i, r in enumerate(routes) for c in r}

    # Greedy merge pass over savings
    for s, ci, cj in savings:
        if time.time() > DEADLINE - MARGIN:
            break
        ri = cust_route.get(ci)
        rj = cust_route.get(cj)
        if ri is None or rj is None or ri == rj:
            continue
        merged = try_merge(ri, rj)
        if merged is not None:
            # ensure merge actually reduces distance
            d_before = 0.0
            for r in (routes[ri], routes[rj]):
                p = 0
                for c in r:
                    d_before += dist(p, c); p = c
                d_before += dist(p, 0)
            d_after = 0.0
            p = 0
            for c in merged:
                d_after += dist(p, c); p = c
            d_after += dist(p, 0)
            if d_after < d_before - 1e-9:
                routes[ri] = merged
                routes[rj] = []
                for c in merged:
                    cust_route[c] = ri

    routes = [r for r in routes if r]

    # ---- Local search: 2-opt within route, relocate, or-opt across routes ----
    def route_dist(r):
        if not r:
            return 0.0
        p = 0; s = 0.0
        for c in r:
            s += dist(p, c); p = c
        s += dist(p, 0)
        return s

    def two_opt(r):
        """In-place 2-opt improvement, returns new list if feasible and shorter."""
        n = len(r)
        if n < 2:
            return r
        best = r[:]
        best_d = route_dist(best)
        improved = True
        while improved and time.time() < DEADLINE - MARGIN:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    cand = best[:i] + list(reversed(best[i:j + 1])) + best[j + 1:]
                    ok, _ = feasible(cand)
                    if not ok:
                        continue
                    d = route_dist(cand)
                    if d < best_d - 1e-9:
                        best = cand
                        best_d = d
                        improved = True
        return best

    def relocate(routes):
        """Move one customer from one route to a better position in another."""
        improved = False
        for ri in range(len(routes)):
            if time.time() > DEADLINE - MARGIN:
                return routes, improved
            for idx in range(len(routes[ri])):
                c = routes[ri][idx]
                src = routes[ri][:idx] + routes[ri][idx + 1:]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    dst = routes[rj]
                    best_pos = None
                    best_gain = None
                    d_src = route_dist(routes[ri])
                    d_dst = route_dist(dst)
                    for pos in range(len(dst) + 1):
                        cand_dst = dst[:pos] + [c] + dst[pos:]
                        ok, _ = feasible(cand_dst)
                        if not ok:
                            continue
                        d2 = route_dist(cand_dst)
                        gain = (d_src + d_dst) - (route_dist(src) + d2)
                        if best_gain is None or gain > best_gain:
                            best_gain = gain
                            best_pos = (cand_dst, gain)
                    if best_pos is not None and best_pos[1] > 1e-9:
                        cand_dst, gain = best_pos
                        routes[ri] = src
                        routes[rj] = cand_dst
                        improved = True
                        break
                if time.time() > DEADLINE - MARGIN:
                    return routes, improved
        return routes, improved

    def or_opt(routes):
        """Move 2-3 consecutive customers to a better position in another route."""
        improved = False
        for L in (2, 3):
            for ri in range(len(routes)):
                if time.time() > DEADLINE - MARGIN:
                    return routes, improved
                for idx in range(len(routes[ri]) - L + 1):
                    seg = routes[ri][idx:idx + L]
                    src = routes[ri][:idx] + routes[ri][idx + L:]
                    for rj in range(len(routes)):
                        if rj == ri:
                            continue
                        dst = routes[rj]
                        best = None
                        d_src = route_dist(routes[ri])
                        d_dst = route_dist(dst)
                        for pos in range(len(dst) + 1):
                            for orient in (seg, list(reversed(seg))):
                                cand_dst = dst[:pos] + orient + dst[pos:]
                                ok, _ = feasible(cand_dst)
                                if not ok:
                                    continue
                                gain = (d_src + d_dst) - (route_dist(src) + route_dist(cand_dst))
                                if best is None or gain > best[1]:
                                    best = (cand_dst, gain)
                        if best is not None and best[1] > 1e-9:
                            routes[ri] = src
                            routes[rj] = best[0]
                            improved = True
                            break
                if time.time() > DEADLINE - MARGIN:
                    return routes, improved
        return routes, improved

    # Polish loop
    for _ in range(40):
        if time.time() > DEADLINE - MARGIN:
            break
        for ri in range(len(routes)):
            if time.time() > DEADLINE - MARGIN:
                break
            routes[ri] = two_opt(routes[ri])
        routes, imp1 = relocate(routes)
        routes, imp2 = or_opt(routes)
        routes = [r for r in routes if r]
        if not imp1 and not imp2:
            break

    # Final safety: verify all customers served exactly once and feasible
    served = []
    for r in routes:
        ok, _ = feasible(r)
        if not ok:
            # fallback: split into singletons
            for c in r:
                routes.append([c])
            r.clear()
            continue
        served.extend(r)
    routes = [r for r in routes if r]
    if sorted(served) != sorted(customers):
        # absolute fallback
        return [[c] for c in customers]
    return routes
# EVOLVE-BLOCK-END
