"""EV routing with time windows: NN construction + feasibility-checked local search."""
# EVOLVE-BLOCK-START
import math
import time

_EPS = 1e-9


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build(instance):
    coords = instance["coords"]
    n = len(coords)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        xi, yi = coords[i]
        row = D[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj = coords[j]
            row[j] = math.hypot(xi - xj, yi - yj)
    return D


def _is_station(instance, node):
    return node in instance["stations_set"]


def _simulate(route, instance, D, stations_set):
    """Return (ok, total_distance)."""
    n = len(route)
    if n == 0:
        return True, 0.0
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    load = 0.0
    battery = float(battery_max)
    t = 0.0
    prev = 0
    total = 0.0

    for node in route:
        d = D[prev][node]
        if d > battery + _EPS:
            return False, total
        battery -= d
        t += d / speed
        total += d
        prev = node
        if node in stations_set:
            # charging stop: no demand, no service, refill
            if battery < 0.0:
                return False, total
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            continue
        # customer
        load += demands[node]
        if load > capacity + _EPS:
            return False, total
        if t < ready[node]:
            t = float(ready[node])
        if t > due[node] + _EPS:
            return False, total
        t += service[node]
        if t > horizon + _EPS:
            return False, total

    # return to depot
    d = D[prev][0]
    if d > battery + _EPS:
        return False, total
    battery -= d
    t += d / speed
    total += d
    if t > horizon + _EPS:
        return False, total
    return True, total


def _nn_route(instance, D, unvisited, stations_set):
    """Build a feasible route via nearest neighbor, inserting charging stops."""
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    route = []
    load = 0.0
    battery = float(battery_max)
    t = 0.0
    cur = 0
    remaining = set(unvisited)
    if not remaining:
        return []

    # choose first customer: nearest feasible
    def feasible_from(node, battery, t, load):
        d = D[cur][node]
        if d > battery + _EPS:
            return False
        nb = battery - d
        nt = t + d / speed
        if node in stations_set:
            return nb >= -_EPS
        nl = load + demands[node]
        if nl > capacity + _EPS:
            return False
        if nt < ready[node]:
            nt = float(ready[node])
        if nt > due[node] + _EPS:
            return False
        nt += service[node]
        return nt <= horizon + _EPS

    # nearest feasible customer
    first = None
    best_d = None
    for c in remaining:
        if not feasible_from(c, battery, t, load):
            continue
        dd = D[cur][c]
        if best_d is None or dd < best_d:
            best_d = dd
            first = c
    if first is None:
        return []

    def do_customer(node):
        nonlocal battery, t, cur, load
        d = D[cur][node]
        battery -= d
        t += d / speed
        cur = node
        load += demands[node]
        if t < ready[node]:
            t = float(ready[node])
        t += service[node]
        route.append(node)
        remaining.discard(node)

    def do_station(node):
        nonlocal battery, t, cur
        d = D[cur][node]
        battery -= d
        t += d / speed
        cur = node
        charge = battery_max - battery
        if charge > 0.0:
            t += charge * recharge
        battery = float(battery_max)
        route.append(node)

    do_customer(first)

    # check we can return to depot; if not, try to charge first
    def can_return():
        d = D[cur][0]
        return d <= battery + _EPS

    while remaining:
        # pick next customer among feasible
        nextc = None
        best_d = None
        for c in remaining:
            if not feasible_from(c, battery, t, load):
                continue
            dd = D[cur][c]
            if best_d is None or dd < best_d:
                best_d = dd
                nextc = c
        if nextc is not None:
            do_customer(nextc)
        else:
            # cannot serve any more; try to close route
            if can_return():
                break
            # try to charge at nearest station then maybe continue or return
            best_st = None
            best_sd = None
            for s in instance["stations"]:
                dd = D[cur][s]
                if dd <= battery + _EPS:
                    if best_sd is None or dd < best_sd:
                        best_sd = dd
                        best_st = s
            if best_st is None:
                # cannot reach any station; break (caller will handle)
                break
            # simulate charging
            battery2 = battery - best_sd
            t2 = t + best_sd / speed
            charge = battery_max - battery2
            t2 += charge * recharge
            battery2 = float(battery_max)
            # after charging, can we return?
            dret = D[best_st][0]
            if dret <= battery2 + _EPS and t2 + dret / speed <= horizon + _EPS:
                do_station(best_st)
                break
            # can we serve any customer after charging?
            found = False
            bestc = None
            bcd = None
            for c in remaining:
                d = D[best_st][c]
                if d > battery2 + _EPS:
                    continue
                nt = t2 + d / speed
                nl = load + demands[c]
                if nl > capacity + _EPS:
                    continue
                if nt < ready[c]:
                    nt = float(ready[c])
                if nt > due[c] + _EPS:
                    continue
                nt += service[c]
                if nt > horizon + _EPS:
                    continue
                if bcd is None or d < bcd:
                    bcd = d
                    bestc = c
            if bestc is not None:
                do_station(best_st)
                # continue loop
                continue
            else:
                do_station(best_st)
                break
    return route


def _insert_station_to_return(route, instance, D, stations_set):
    """Try to make a route returnable by inserting a charging station before the end.
    Returns new route or None."""
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    if not route:
        return route
    # simulate to get state at each step
    # We'll try inserting a station at position p (0..len(route)) meaning after route[p-1]
    # Simplest: try inserting nearest station that lets us return.
    # Simulate full route to depot to find where battery runs out.
    # We'll just try: for each station, insert at the position right before the point
    # where we can no longer continue. Brute force over positions is fine (small).
    n = len(route)
    for s in instance["stations"]:
        for p in range(0, n + 1):
            cand = route[:p] + [s] + route[p:]
            ok, _ = _simulate(cand, instance, D, stations_set)
            if ok:
                return cand
    return None


def _construct(instance, D, stations_set, deadline):
    customers = list(range(1, instance["n_customers"] + 1))
    # sort by due for better construction
    customers.sort(key=lambda v: (instance["due"][v], v))
    unvisited = set(customers)
    routes = []
    while unvisited:
        if time.time() >= deadline:
            break
        r = _nn_route(instance, D, unvisited, stations_set)
        if not r:
            # cannot build any route from remaining; take the single nearest
            c = min(unvisited, key=lambda v: (D[0][v], v))
            r = [c]
        # ensure returnable
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            fixed = _insert_station_to_return(r, instance, D, stations_set)
            if fixed is not None:
                r = fixed
            else:
                # fallback: keep only customers that are individually feasible
                keep = []
                for node in r:
                    if node in stations_set:
                        continue
                    single = [node]
                    ok1, _ = _simulate(single, instance, D, stations_set)
                    if ok1:
                        keep.append(node)
                if keep:
                    r = keep
                else:
                    r = [r[0] if r else c]
        unvisited -= set(r)
        routes.append(r)
    return routes


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0][route[0]]
    for i in range(1, len(route)):
        total += D[route[i - 1]][route[i]]
    total += D[route[-1]][0]
    return total


def _total_distance(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _routes_valid(routes, instance, D, stations_set):
    seen = set()
    for r in routes:
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            return False
        for node in r:
            if node in stations_set or node == 0:
                continue
            if node in seen:
                return False
            seen.add(node)
    n_cust = instance["n_customers"]
    if len(seen) != n_cust:
        return False
    return True


def _swap2opt(routes, instance, D, stations_set, deadline):
    """2-opt within each route, plus relocate between routes. Returns improved routes."""
    n_routes = len(routes)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # 2-opt within each route
        for ri in range(n_routes):
            r = routes[ri]
            m = len(r)
            if m < 3:
                continue
            for i in range(0, m - 1):
                for j in range(i + 1, m):
                    cand = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                    ok, _ = _simulate(cand, instance, D, stations_set)
                    if ok and _route_dist(cand, D) < _route_dist(r, D) - _EPS:
                        routes[ri] = cand
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        # relocate a node from one route to another
        for a in range(n_routes):
            ra = routes[a]
            for pos in range(len(ra)):
                node = ra[pos]
                if node in stations_set:
                    continue
                rest_a = ra[:pos] + ra[pos + 1:]
                for b in range(n_routes):
                    if b == a:
                        continue
                    rb = routes[b]
                    for q in range(len(rb) + 1):
                        cand_b = rb[:q] + [node] + rb[q:]
                        okb, _ = _simulate(cand_b, instance, D, stations_set)
                        oka, _ = _simulate(rest_a, instance, D, stations_set)
                        if oka and okb:
                            new_total = _route_dist(cand_b, D) + _route_dist(rest_a, D)
                            old_total = _route_dist(ra, D) + _route_dist(rb, D)
                            if new_total < old_total - _EPS:
                                routes[a] = rest_a
                                routes[b] = cand_b
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    # drop empty routes
    routes = [r for r in routes if r]
    return routes


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    stations_set = set(instance["stations"])
    instance["stations_set"] = stations_set
    D = _build(instance)

    routes = _construct(instance, D, stations_set, deadline)

    # ensure all customers covered; add any missing as singletons
    covered = set()
    for r in routes:
        for node in r:
            if node not in stations_set and node != 0:
                covered.add(node)
    missing = [v for v in range(1, instance["n_customers"] + 1) if v not in covered]
    for v in missing:
        routes.append([v])

    # local search under time budget
    routes = _swap2opt(routes, instance, D, stations_set, deadline)

    # final validity check; if invalid, fall back to singletons
    if not _routes_valid(routes, instance, D, stations_set):
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    return routes
# EVOLVE-BLOCK-END
