"""EV routing with time windows: cluster-first + parallel local search."""
# EVOLVE-BLOCK-START
import math
import time
import random
import heapq

_EPS = 1e-9


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


def _simulate(route, instance, D, stations_set):
    """Return (ok, total_distance)."""
    if not route:
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
            if battery < 0.0:
                return False, total
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            continue
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

    d = D[prev][0]
    if d > battery + _EPS:
        return False, total
    battery -= d
    t += d / speed
    total += d
    if t > horizon + _EPS:
        return False, total
    return True, total


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


def _cluster_customers(instance, D, n_clusters):
    """Cluster customers using k-means on 2D coords."""
    coords = instance["coords"]
    customers = list(range(1, instance["n_customers"] + 1))
    n_cust = len(customers)
    
    if n_clusters >= n_cust:
        return [[c] for c in customers]
    
    # Initialize centers using k-means++
    rng = random.Random(42)
    centers = [list(coords[customers[rng.randint(0, n_cust - 1)]])]
    for _ in range(1, n_clusters):
        min_dists = []
        for c in customers:
            min_d = min(math.hypot(coords[c][0] - cx[0], coords[c][1] - cx[1]) for cx in centers)
            min_dists.append(min_d)
        total = sum(min_dists)
        if total == 0:
            centers.append(list(coords[customers[rng.randint(0, n_cust - 1)]]))
            continue
        r = rng.random() * total
        cum = 0
        for i, md in enumerate(min_dists):
            cum += md
            if cum >= r:
                centers.append(list(coords[customers[i]]))
                break
        else:
            centers.append(list(coords[customers[-1]]))
    
    # K-means iterations
    clusters = [[] for _ in range(n_clusters)]
    for _ in range(20):
        # Assign
        clusters = [[] for _ in range(n_clusters)]
        for c in customers:
            best_k = 0
            best_d = float('inf')
            for k in range(n_clusters):
                d = math.hypot(coords[c][0] - centers[k][0], coords[c][1] - centers[k][1])
                if d < best_d:
                    best_d = d
                    best_k = k
            clusters[best_k].append(c)
        
        # Update centers
        changed = False
        for k in range(n_clusters):
            if not clusters[k]:
                continue
            sx = sum(coords[c][0] for c in clusters[k]) / len(clusters[k])
            sy = sum(coords[c][1] for c in clusters[k]) / len(clusters[k])
            if abs(sx - centers[k][0]) > 1e-6 or abs(sy - centers[k][1]) > 1e-6:
                changed = True
            centers[k] = [sx, sy]
        if not changed:
            break
    
    # Merge empty clusters
    clusters = [c for c in clusters if c]
    return clusters


def _nn_route_from_cluster(instance, D, cluster, stations_set):
    """Build a feasible route for a cluster using nearest neighbor."""
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    
    if not cluster:
        return []
    
    route = []
    load = 0.0
    battery = float(battery_max)
    t = 0.0
    cur = 0
    remaining = set(cluster)
    
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
    
    # Pick first customer
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
        # Try with a station first
        best_st = None
        best_sd = None
        for s in instance["stations"]:
            dd = D[cur][s]
            if dd <= battery + _EPS:
                if best_sd is None or dd < best_sd:
                    best_sd = dd
                    best_st = s
        if best_st is not None:
            d = D[cur][best_st]
            battery -= d
            t += d / speed
            cur = best_st
            charge = battery_max - battery
            if charge > 0.0:
                t += charge * recharge
            battery = float(battery_max)
            route.append(best_st)
            for c in remaining:
                if not feasible_from(c, battery, t, load):
                    continue
                dd = D[cur][c]
                if best_d is None or dd < best_d:
                    best_d = dd
                    first = c
        if first is None:
            return [list(remaining)[0]]
    
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
    
    def can_return():
        d = D[cur][0]
        return d <= battery + _EPS
    
    while remaining:
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
            if can_return():
                break
            best_st = None
            best_sd = None
            for s in instance["stations"]:
                dd = D[cur][s]
                if dd <= battery + _EPS:
                    if best_sd is None or dd < best_sd:
                        best_sd = dd
                        best_st = s
            if best_st is None:
                break
            battery2 = battery - best_sd
            t2 = t + best_sd / speed
            charge = battery_max - battery2
            t2 += charge * recharge
            battery2 = float(battery_max)
            dret = D[best_st][0]
            if dret <= battery2 + _EPS and t2 + dret / speed <= horizon + _EPS:
                do_station(best_st)
                break
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
                continue
            else:
                do_station(best_st)
                break
    return route


def _try_add_station(route, instance, D, stations_set):
    """Try inserting a charging station to make route feasible."""
    n = len(route)
    best = None
    best_d = float('inf')
    for s in instance["stations"]:
        for p in range(n + 1):
            cand = route[:p] + [s] + route[p:]
            ok, dist = _simulate(cand, instance, D, stations_set)
            if ok and dist < best_d:
                best_d = dist
                best = cand
    return best


def _construct_cluster(instance, D, stations_set, n_clusters):
    """Cluster-based construction."""
    clusters = _cluster_customers(instance, D, n_clusters)
    routes = []
    for cluster in clusters:
        r = _nn_route_from_cluster(instance, D, cluster, stations_set)
        if not r:
            continue
        ok, _ = _simulate(r, instance, D, stations_set)
        if not ok:
            fixed = _try_add_station(r, instance, D, stations_set)
            if fixed is not None:
                r = fixed
            else:
                # Split into singletons
                for node in r:
                    if node in stations_set:
                        continue
                    single = [node]
                    ok1, _ = _simulate(single, instance, D, stations_set)
                    if ok1:
                        routes.append(single)
                    else:
                        fixed1 = _try_add_station(single, instance, D, stations_set)
                        if fixed1 is not None:
                            routes.append(fixed1)
        else:
            routes.append(r)
    return routes


def _construct_nn(instance, D, stations_set, deadline, rng):
    """Standard NN construction."""
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    
    customers = list(range(1, instance["n_customers"] + 1))
    customers.sort(key=lambda v: (instance["due"][v], v))
    unvisited = set(customers)
    routes = []
    
    while unvisited:
        if time.time() >= deadline:
            break
        route = []
        load = 0.0
        battery = float(battery_max)
        t = 0.0
        cur = 0
        remaining = set(unvisited)
        
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
            c = min(unvisited, key=lambda v: (D[0][v], v))
            route = [c]
            unvisited.discard(c)
            routes.append(route)
            continue
        
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
        
        def can_return():
            d = D[cur][0]
            return d <= battery + _EPS
        
        while remaining:
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
                if can_return():
                    break
                best_st = None
                best_sd = None
                for s in instance["stations"]:
                    dd = D[cur][s]
                    if dd <= battery + _EPS:
                        if best_sd is None or dd < best_sd:
                            best_sd = dd
                            best_st = s
                if best_st is None:
                    break
                battery2 = battery - best_sd
                t2 = t + best_sd / speed
                charge = battery_max - battery2
                t2 += charge * recharge
                battery2 = float(battery_max)
                dret = D[best_st][0]
                if dret <= battery2 + _EPS and t2 + dret / speed <= horizon + _EPS:
                    do_station(best_st)
                    break
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
                    continue
                else:
                    do_station(best_st)
                    break
        
        unvisited -= set(route)
        routes.append(route)
    
    return routes


def _oropt(routes, instance, D, stations_set, deadline):
    """Or-opt moves."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_routes = len(routes)
        for ri in range(n_routes):
            r = routes[ri]
            m = len(r)
            if m < 2:
                continue
            for seg_len in [1, 2, 3]:
                for i in range(m - seg_len + 1):
                    seg = r[i:i + seg_len]
                    if any(n in stations_set for n in seg):
                        continue
                    rest = r[:i] + r[i + seg_len:]
                    for j in range(len(rest) + 1):
                        if j == i or j == i + seg_len:
                            continue
                        cand = rest[:j] + seg + rest[j:]
                        ok, _ = _simulate(cand, instance, D, stations_set)
                        if ok and _route_dist(cand, D) < _route_dist(r, D) - _EPS:
                            routes[ri] = cand
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            for seg_len in [1, 2, 3]:
                for i in range(m - seg_len + 1):
                    seg = r[i:i + seg_len]
                    if any(n in stations_set for n in seg):
                        continue
                    rest_a = r[:i] + r[i + seg_len:]
                    for bj in range(n_routes):
                        if bj == ri:
                            continue
                        rb = routes[bj]
                        for q in range(len(rb) + 1):
                            cand_b = rb[:q] + seg + rb[q:]
                            okb, _ = _simulate(cand_b, instance, D, stations_set)
                            oka, _ = _simulate(rest_a, instance, D, stations_set)
                            if oka and okb:
                                new_total = _route_dist(cand_b, D) + _route_dist(rest_a, D)
                                old_total = _route_dist(r, D) + _route_dist(rb, D)
                                if new_total < old_total - _EPS:
                                    routes[ri] = rest_a
                                    routes[bj] = cand_b
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
    routes = [r for r in routes if r]
    return routes


def _two_opt(routes, instance, D, stations_set, deadline):
    """2-opt within each route."""
    n_routes = len(routes)
    improved = True
    while improved and time.time() < deadline:
        improved = False
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
    routes = [r for r in routes if r]
    return routes


def _relocate(routes, instance, D, stations_set, deadline):
    """Relocate a node from one route to another."""
    n_routes = len(routes)
    improved = True
    while improved and time.time() < deadline:
        improved = False
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
    routes = [r for r in routes if r]
    return routes


def _local_search(routes, instance, D, stations_set, deadline):
    """Apply multiple local search operators."""
    t0 = time.time()
    remaining = deadline - t0
    third = remaining / 3
    
    routes = _oropt(routes, instance, D, stations_set, t0 + third)
    routes = _two_opt(routes, instance, D, stations_set, t0 + 2 * third)
    routes = _relocate(routes, instance, D, stations_set, deadline)
    
    if time.time() < deadline - 5:
        routes = _oropt(routes, instance, D, stations_set, deadline)
    
    return routes


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0
    stations_set = set(instance["stations"])
    D = _build(instance)
    
    rng = random.Random(42)
    
    # Try multiple construction strategies
    best_routes = None
    best_dist = float('inf')
    
    # Strategy 1: Cluster with 5 clusters
    routes1 = _construct_cluster(instance, D, stations_set, 5)
    if _routes_valid(routes1, instance, D, stations_set):
        d1 = _total_distance(routes1, D)
        if d1 < best_dist:
            best_dist = d1
            best_routes = [r[:] for r in routes1]
    
    # Strategy 2: Cluster with 4 clusters
    routes2 = _construct_cluster(instance, D, stations_set, 4)
    if _routes_valid(routes2, instance, D, stations_set):
        d2 = _total_distance(routes2, D)
        if d2 < best_dist:
            best_dist = d2
            best_routes = [r[:] for r in routes2]
    
    # Strategy 3: Standard NN
    routes3 = _construct_nn(instance, D, stations_set, deadline, rng)
    if _routes_valid(routes3, instance, D, stations_set):
        d3 = _total_distance(routes3, D)
        if d3 < best_dist:
            best_dist = d3
            best_routes = [r[:] for r in routes3]
    
    if best_routes is None:
        best_routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    
    # Local search
    best_routes = _local_search(best_routes, instance, D, stations_set, deadline)
    
    # Final validity check
    if not _routes_valid(best_routes, instance, D, stations_set):
        best_routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    
    return best_routes
# EVOLVE-BLOCK-END
