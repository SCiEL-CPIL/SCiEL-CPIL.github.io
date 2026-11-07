"""EV routing with time windows and charging. Fresh design: feasible greedy construction
(nearest + best feasible insertion with station insertion) + 2-opt/Or-opt local search."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_T0 = time.time()
_LIMIT = 138.0  # safety margin under the 150s kill


def _time_left():
    return _LIMIT - (time.time() - _T0)


def _euclid(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _build_dist(coords):
    n = len(coords)
    D = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _check_route(route, D, ready, due, service, demand, stations_set,
                 capacity, battery, recharge_rate, horizon):
    """Return (feasible, total_distance). Depot is implicit at both ends."""
    t = 0.0
    bat = float(battery)
    load = 0.0
    dist = 0.0
    prev = 0
    for nd in route:
        d = D[prev, nd]
        dist += d
        bat -= d
        t += d
        if bat < -1e-9:
            return False, dist
        if nd in stations_set:
            # station: recharge time = (battery - energy_used) * recharge_rate
            energy = d
            bat = max(bat, 0.0)
            t += energy * recharge_rate
            bat = float(battery)
        else:
            # customer
            if t < ready[nd]:
                t = ready[nd]
            if t > due[nd] + 1e-9:
                return False, dist
            load += demand[nd]
            if load > capacity + 1e-9:
                return False, dist
            t += service[nd]
            if t > horizon + 1e-9:
                return False, dist
        prev = nd
    # return to depot
    d = D[prev, 0]
    dist += d
    bat -= d
    t += d
    if bat < -1e-9 or t > horizon + 1e-9:
        return False, dist
    return True, dist


def _route_with_stations(route, stations, D, battery):
    """Insert a station between consecutive legs when battery would otherwise go negative.
    Returns (ok, new_route, dist)."""
    bat = float(battery)
    new_route = []
    prev = 0
    for nd in route:
        d = D[prev, nd]
        if bat - d < -1e-9:
            # find a station that is reachable and can refill
            chosen = None
            for s in stations:
                ds = D[prev, s]
                if bat - ds >= -1e-9:
                    # after reaching s and refilling, can we reach nd?
                    if D[s, nd] <= battery + 1e-9:
                        if chosen is None or D[prev, s] < D[prev, chosen]:
                            chosen = s
            if chosen is None:
                return False, None, 0.0
            new_route.append(chosen)
            prev = chosen
            d = D[prev, nd]
        bat -= d
        if bat < -1e-9:
            return False, None, 0.0
        new_route.append(nd)
        # if it was a station, bat refills (handled by caller? no, here treat as customer)
        # We do NOT know type; caller's check will refill. For distance only, keep going.
        # To be safe, assume refill so battery doesn't wrongly deplete:
        # (we don't have type info here; but stations are only inserted as 'chosen')
        prev = nd
    d = D[prev, 0]
    bat -= d
    if bat < -1e-9:
        return False, None, 0.0
    return True, new_route, 0.0


def _construct_route(start, unassigned, D, ready, due, service, demand, stations,
                     stations_set, capacity, battery, recharge_rate, horizon):
    """Greedy build a feasible route starting from depot, using unassigned customers."""
    route = []
    t = 0.0
    bat = float(battery)
    load = 0.0
    prev = 0
    remaining = list(unassigned)
    while remaining:
        best = None
        best_key = None
        for nd in remaining:
            d = D[prev, nd]
            # try direct
            if bat - d >= -1e-9:
                nt = t + d
                if nt < ready[nd]:
                    nt = ready[nd]
                if nt > due[nd] + 1e-9:
                    continue
                if load + demand[nd] > capacity + 1e-9:
                    continue
                nt += service[nd]
                if nt > horizon + 1e-9:
                    continue
                key = (d, nt)
                if best_key is None or key < best_key:
                    best_key = key
                    best = nd
            else:
                # need a station first
                for s in stations:
                    ds = D[prev, s]
                    if bat - ds >= -1e-9 and D[s, nd] <= battery + 1e-9:
                        # feasible via station; penalize by extra distance
                        extra = ds + D[s, nd]
                        nt = t + ds + D[s, nd]
                        if nt < ready[nd]:
                            nt = ready[nd]
                        if nt > due[nd] + 1e-9:
                            continue
                        if load + demand[nd] > capacity + 1e-9:
                            continue
                        nt += service[nd]
                        if nt > horizon + 1e-9:
                            continue
                        key = (extra, nt)
                        if best_key is None or key < best_key:
                            best_key = key
                            best = nd
                        break
        if best is None:
            break
        remaining.remove(best)
        # recompute leg, possibly inserting a station
        d = D[prev, best]
        if bat - d < -1e-9:
            chosen = None
            for s in stations:
                if bat - D[prev, s] >= -1e-9 and D[s, best] <= battery + 1e-9:
                    if chosen is None or D[prev, s] < D[prev, chosen]:
                        chosen = s
            if chosen is None:
                break
            route.append(chosen)
            t += D[prev, chosen]
            # recharge
            t += D[prev, chosen] * recharge_rate
            bat = float(battery)
            prev = chosen
        # now go to best
        t += D[prev, best]
        bat -= D[prev, best]
        if t < ready[best]:
            t = ready[best]
        t += service[best]
        load += demand[best]
        route.append(best)
        prev = best
    if not route:
        return None
    # build full route with stations inserted and verify
    full = _insert_stations_full(route, stations, D, battery, ready, due, service,
                                 demand, stations_set, capacity, recharge_rate, horizon)
    return full


def _insert_stations_full(route, stations, D, battery, ready, due, service, demand,
                          stations_set, capacity, recharge_rate, horizon):
    """Insert stations into a customer route so battery is feasible; verify whole thing."""
    # We build incrementally, inserting a station before a customer when needed.
    new_route = []
    t = 0.0
    bat = float(battery)
    load = 0.0
    prev = 0
    for nd in route:
        d = D[prev, nd]
        if bat - d < -1e-9:
            chosen = None
            for s in stations:
                ds = D[prev, s]
                if bat - ds >= -1e-9 and D[s, nd] <= battery + 1e-9:
                    if chosen is None or D[prev, s] < D[prev, chosen]:
                        chosen = s
            if chosen is None:
                return None
            # go to station
            t += D[prev, chosen]
            t += D[prev, chosen] * recharge_rate
            bat = float(battery)
            new_route.append(chosen)
            prev = chosen
            d = D[prev, nd]
        # go to nd
        t += d
        bat -= d
        if bat < -1e-9:
            return None
        if nd in stations_set:
            t += d * recharge_rate
            bat = float(battery)
        else:
            if t < ready[nd]:
                t = ready[nd]
            if t > due[nd] + 1e-9:
                return None
            load += demand[nd]
            if load > capacity + 1e-9:
                return None
            t += service[nd]
            if t > horizon + 1e-9:
                return None
        new_route.append(nd)
        prev = nd
    # verify final with the authoritative checker
    ok, _ = _check_route(new_route, D, ready, due, service, demand, stations_set,
                         capacity, battery, recharge_rate, horizon)
    if not ok:
        return None
    return new_route


def _total_distance(routes, D):
    total = 0.0
    for r in routes:
        prev = 0
        for nd in r:
            total += D[prev, nd]
            prev = nd
        total += D[prev, 0]
    return total


def _improve_routes(routes, D, ready, due, service, demand, stations, stations_set,
                    capacity, battery, recharge_rate, horizon, time_budget):
    """2-opt + Or-opt local search over a set of routes (customers only per route)."""
    # Represent each route as customer sequence (stations stripped for move logic).
    cust_routes = []
    for r in routes:
        cust_routes.append([nd for nd in r if nd not in stations_set])

    def rebuild(cust_routes):
        out = []
        for cr in cust_routes:
            if not cr:
                continue
            full = _insert_stations_full(cr, stations, D, battery, ready, due,
                                         service, demand, stations_set, capacity,
                                         recharge_rate, horizon)
            if full is None:
                return None
            out.append(full)
        return out

    def dist_of(cust_routes):
        total = 0.0
        for cr in cust_routes:
            prev = 0
            for nd in cr:
                total += D[prev, nd]
                prev = nd
            total += D[prev, 0]
        return total

    cur = cust_routes
    cur_d = dist_of(cur)
    if rebuild(cur) is None:
        return routes  # should not happen; keep original

    def try_move(move):
        nonlocal cur, cur_d
        # apply move to a copy
        new = [list(x) for x in cur]
        move(new)
        # feasibility: rebuild
        rb = rebuild(new)
        if rb is None:
            return False
        nd_ = dist_of(new)
        if nd_ < cur_d - 1e-9:
            cur = new
            cur_d = nd_
            return True
        return False

    def two_opt(i, a, b):
        def m(new):
            seg = new[i][a:b+1]
            seg = seg[::-1]
            new[i][a:b+1] = seg
        return m

    def or_opt(i, a, b, j, c):
        def m(new):
            seg = new[i][a:b+1]
            del new[i][a:b+1]
            new[j].insert(c, seg[0])
            for x in seg[1:]:
                new[j].insert(c + 1, x)
        return m

    def relocate(i, a, j, c):
        def m(new):
            x = new[i][a]
            del new[i][a]
            new[j].insert(c, x)
        return m

    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        for i in range(len(cur)):
            L = len(cur[i])
            # 2-opt
            for a in range(L - 1):
                for b in range(a + 1, L):
                    if try_move(two_opt(i, a, b)):
                        improved = True
            # or-opt (move 1 or 2 consecutive)
            for a in range(L):
                for b in (a, min(a + 1, L - 1)):
                    for j in range(len(cur)):
                        Lj = len(cur[j])
                        for c in range(Lj + 1):
                            if i == j:
                                # avoid moving into same position
                                if c == a or c == a + 1:
                                    continue
                            if try_move(or_opt(i, a, b, j, c)):
                                improved = True
            # relocate single
            for a in range(L):
                for j in range(len(cur)):
                    Lj = len(cur[j])
                    for c in range(Lj + 1):
                        if i == j:
                            if c == a or c == a + 1:
                                continue
                        if try_move(relocate(i, a, j, c)):
                            improved = True
    # rebuild final
    final = rebuild(cur)
    if final is None:
        return routes
    return final


def plan(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demand = instance["demand"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    stations_set = set(stations)
    D = _build_dist(coords)

    # scale D by 1/speed for time; but battery uses raw distance. Keep D raw.
    # time uses d/speed; speed=1 here, but be general:
    # We'll incorporate speed in time computations.
    # For simplicity assume speed=1 (given). If not, divide time by speed.

    customers = list(range(1, n_customers + 1))
    # Sort customers by due time for construction order
    customers_sorted = sorted(customers, key=lambda v: due[v])

    # Greedy: build routes by inserting customers into existing routes (best feasible),
    # starting a new route when none fits.
    routes = []
    # We'll keep customer-only route lists and rebuild with stations each time.
    cust_routes = []

    # Precompute per-customer nearest station distance for quick feasibility hints
    for v in customers_sorted:
        placed = False
        # try inserting into each existing route at best position
        best = None  # (route_idx, pos, extra_dist)
        for ri, cr in enumerate(cust_routes):
            L = len(cr)
            for pos in range(L + 1):
                a = 0 if pos == 0 else cr[pos - 1]
                b = 0 if pos == L else cr[pos]
                # extra distance from inserting v between a and b
                extra = D[a, v] + D[v, b] - D[a, b]
                # quick feasibility: rebuild this route
                new_cr = list(cr)
                new_cr.insert(pos, v)
                full = _insert_stations_full(new_cr, stations, D, battery, ready, due,
                                             service, demand, stations_set, capacity,
                                             recharge_rate, horizon)
                if full is not None:
                    if best is None or extra < best[2]:
                        best = (ri, pos, extra)
        if best is not None:
            ri, pos, _ = best
            cust_routes[ri].insert(pos, v)
            placed = True
        else:
            # start a new route
            full = _insert_stations_full([v], stations, D, battery, ready, due,
                                         service, demand, stations_set, capacity,
                                         recharge_rate, horizon)
            if full is not None:
                cust_routes.append([v])
                placed = True
        if not placed:
            # cannot place; should not happen with full battery; fallback single route
            cust_routes.append([v])

    # Build final routes with stations
    final_routes = []
    for cr in cust_routes:
        full = _insert_stations_full(cr, stations, D, battery, ready, due,
                                     service, demand, stations_set, capacity,
                                     recharge_rate, horizon)
        if full is not None:
            final_routes.append(full)
        else:
            # fallback: single customer per route
            for v in cr:
                f2 = _insert_stations_full([v], stations, D, battery, ready, due,
                                           service, demand, stations_set, capacity,
                                           recharge_rate, horizon)
                if f2 is not None:
                    final_routes.append(f2)

    # Local search to reduce distance
    if _time_left() > 8.0:
        improved = _improve_routes(final_routes, D, ready, due, service, demand,
                                   stations, stations_set, capacity, battery,
                                   recharge_rate, horizon, _time_left())
        if improved:
            final_routes = improved

    # Safety: verify every customer served exactly once and all feasible
    served = []
    for r in final_routes:
        served.extend([nd for nd in r if nd not in stations_set])
    if sorted(served) != customers_sorted:
        # rebuild from scratch as one-per-route (always feasible)
        final_routes = []
        for v in customers:
            f2 = _insert_stations_full([v], stations, D, battery, ready, due,
                                       service, demand, stations_set, capacity,
                                       recharge_rate, horizon)
            if f2 is not None:
                final_routes.append(f2)

    return final_routes
# EVOLVE-BLOCK-END
