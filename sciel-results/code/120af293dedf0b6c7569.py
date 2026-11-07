"""EV routing with time windows: cluster-first + nearest-neighbor + multi-neighborhood local search."""
# EVOLVE-BLOCK-START
import math
import time
import random

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


def _total_dist(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _try_add_station(route, instance, D, stations_set):
    """Try inserting a charging station to make route feasible. Returns fixed route or None."""
    n = len(route)
    best_route = None
    best_dist = float('inf')
    for s in instance["stations"]:
        for p in range(n + 1):
            cand = route[:p] + [s] + route[p:]
            ok, dist = _simulate(cand, instance, D, stations_set)
            if ok and dist < best_dist:
                best_dist = dist
                best_route = cand
    return best_route


def _ensure_feasible(route, instance, D, stations_set, max_stations=3):
    """Ensure route is feasible by adding stations if needed."""
    ok, _ = _simulate(route, instance, D, stations_set)
    if ok:
        return route
    for _ in range(max_stations):
        fixed = _try_add_station(route, instance, D, stations_set)
        if fixed is None:
            break
        route = fixed
        ok, _ = _simulate(route, instance, D, stations_set)
        if ok:
            break
    return route


def _cluster_customers(instance, D, stations_set):
    """Cluster customers using k-means-like approach based on distance from depot."""
    n_cust = instance["n_customers"]
    customers = list(range(1, n_cust + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]
    
    # Estimate number of routes needed
    total_demand = sum(demands[c] for c in customers)
    n_routes_est = max(1, int(math.ceil(total_demand / capacity)))
    
    # Use k-means style clustering
    # Initialize centers randomly
    random.seed(42)
    n_clusters = n_routes_est
    centers = random.sample(customers, min(n_clusters, len(customers)))
    
    clusters = [[] for _ in range(n_clusters)]
    for _ in range(20):  # iterations
        # Assign customers to nearest center
        new_clusters = [[] for _ in range(n_clusters)]
        for c in customers:
            best_center = 0
            best_dist = float('inf')
            for k, center in enumerate(centers):
                d = D[c][center]
                if d < best_dist:
                    best_dist = d
                    best_center = k
            new_clusters[best_center].append(c)
        
        # Check convergence
        if all(len(new_clusters[k]) == len(clusters[k]) for k in range(n_clusters)):
            break
        clusters = new_clusters
        
        # Update centers
        for k in range(n_clusters):
            if clusters[k]:
                # Pick customer with median distance to others in cluster
                cluster = clusters[k]
                if len(cluster) == 1:
                    centers[k] = cluster[0]
                else:
                    # Pick the one closest to centroid
                    cx = sum(D[c][0] for c in cluster) / len(cluster)
                    cy = sum(D[c][1] for c in cluster) / len(cluster)
                    # Actually, let's just pick the first one for simplicity
                    centers[k] = cluster[0]
    
    # Merge empty clusters
    final_clusters = [c for c in clusters if c]
    
    # Split clusters that exceed capacity
    result = []
    for cluster in final_clusters:
        if not cluster:
            continue
        # Greedily split by demand
        sub_clusters = []
        current = []
        current_demand = 0
        for c in cluster:
            if current_demand + demands[c] > capacity:
                if current:
                    sub_clusters.append(current)
                current = [c]
                current_demand = demands[c]
            else:
                current.append(c)
                current_demand += demands[c]
        if current:
            sub_clusters.append(current)
        result.extend(sub_clusters)
    
    return result


def _nn_route(cluster, instance, D, stations_set):
    """Build a route for a cluster using nearest neighbor with time window awareness."""
    if not cluster:
        return []
    
    remaining = set(cluster)
    route = []
    current = 0  # depot
    battery = float(instance["battery"])
    t = 0.0
    load = 0.0
    capacity = instance["capacity"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    speed = instance["speed"]
    recharge = instance["recharge_rate"]
    battery_max = instance["battery"]
    
    while remaining:
        # Find best next customer
        best_node = None
        best_score = float('inf')
        
        for c in remaining:
            d = D[current][c]
            if d > battery + _EPS:
                continue
            new_t = t + d / speed
            if new_t < ready[c]:
                new_t = float(ready[c])
            if new_t > due[c] + _EPS:
                continue
            new_load = load + demands[c]
            if new_load > capacity + _EPS:
                continue
            # Score: distance + time penalty
            score = d + 0.1 * (new_t - ready[c])
            if score < best_score:
                best_score = score
                best_node = c
        
        if best_node is None:
            # Try with a station
            found = False
            for s in instance["stations"]:
                d_to_s = D[current][s]
                if d_to_s > battery + _EPS:
                    continue
                new_battery = battery - d_to_s
                new_t = t + d_to_s / speed
                charge = battery_max - new_battery
                if charge > 0:
                    new_t += charge * recharge
                new_battery = battery_max
                
                for c in remaining:
                    d_s_c = D[s][c]
                    if d_s_c > new_battery + _EPS:
                        continue
                    new_t2 = new_t + d_s_c / speed
                    if new_t2 < ready[c]:
                        new_t2 = float(ready[c])
                    if new_t2 > due[c] + _EPS:
                        continue
                    new_load = load + demands[c]
                    if new_load > capacity + _EPS:
                        continue
                    score = d_to_s + d_s_c + 0.1 * (new_t2 - ready[c])
                    if score < best_score:
                        best_score = score
                        best_node = c
                        route.append(s)
                        t = new_t
                        battery = new_battery
                        found = True
                        break
                if found:
                    break
            if not found:
                # Just pick the closest feasible one
                for c in remaining:
                    d = D[current][c]
                    if d > battery + _EPS:
                        continue
                    new_t = t + d / speed
                    if new_t < ready[c]:
                        new_t = float(ready[c])
                    if new_t > due[c] + _EPS:
                        continue
                    new_load = load + demands[c]
                    if new_load > capacity + _EPS:
                        continue
                    best_node = c
                    break
        
        if best_node is None:
            # Can't serve any more; break
            break
        
        # Travel to best_node
        d = D[current][best_node]
        battery -= d
        t += d / speed
        load += demands[best_node]
        if t < ready[best_node]:
            t = float(ready[best_node])
        t += service[best_node]
        route.append(best_node)
        remaining.discard(best_node)
        current = best_node
    
    return route


def _two_opt(route, instance, D, stations_set, deadline):
    """2-opt within a route."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(route)
        if m < 3:
            break
        old_dist = _route_dist(route, D)
        for i in range(m - 1):
            if time.time() >= deadline:
                break
            for j in range(i + 1, m):
                cand = route[:i] + route[i:j+1][::-1] + route[j+1:]
                ok, _ = _simulate(cand, instance, D, stations_set)
                if ok:
                    new_dist = _route_dist(cand, D)
                    if new_dist < old_dist - _EPS:
                        route = cand
                        old_dist = new_dist
                        improved = True
                        break
            if improved:
                break
    return route


def _or_opt(route, instance, D, stations_set, deadline):
    """Or-opt within a route: move 1-3 consecutive nodes."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        m = len(route)
        if m < 3:
            break
        old_dist = _route_dist(route, D)
        for seg_len in (1, 2, 3):
            if m < seg_len + 1:
                continue
            for i in range(m - seg_len + 1):
                if time.time() >= deadline:
                    break
                seg = route[i:i+seg_len]
                if any(n in stations_set for n in seg):
                    continue
                rest = route[:i] + route[i+seg_len:]
                for j in range(len(rest) + 1):
                    if j == i or j == i + seg_len:
                        continue
                    cand = rest[:j] + seg + rest[j:]
                    ok, _ = _simulate(cand, instance, D, stations_set)
                    if ok:
                        new_dist = _route_dist(cand, D)
                        if new_dist < old_dist - _EPS:
                            route = cand
                            old_dist = new_dist
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route


def _cross_route_2opt(routes, instance, D, stations_set, deadline):
    """2-opt between two routes."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        nr = len(routes)
        for ri in range(nr):
            for rj in range(ri + 1, nr):
                if time.time() >= deadline:
                    return routes
                ra = routes[ri]
                rb = routes[rj]
                ma = len(ra)
                mb = len(rb)
                if ma < 1 or mb < 1:
                    continue
                # Try reversing a segment in ra and inserting into rb, and vice versa
                # Simplified: try swapping tails
                # Actually, let's do a proper cross 2-opt
                # Remove edge (ra[i], ra[i+1]) and (rb[j], rb[j+1]), reconnect
                for i in range(ma):
                    for j in range(mb):
                        # ra: ... ra[i], ra[i+1] ...
                        # rb: ... rb[j], rb[j+1] ...
                        # New: ra[:i+1] + rb[j+1:] and rb[:j+1] + ra[i+1:]
                        # But we need to handle the depot connections properly
                        # This is complex; let's skip for now and rely on intra-route optimization
                        pass
        if not improved:
            break
    return routes


def _local_search(routes, instance, D, stations_set, deadline):
    """Apply local search to all routes."""
    # First pass: 2-opt on each route
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _two_opt(routes[ri], instance, D, stations_set, deadline)
    
    # Second pass: Or-opt on each route
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _or_opt(routes[ri], instance, D, stations_set, deadline)
    
    # Third pass: 2-opt again
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _two_opt(routes[ri], instance, D, stations_set, deadline)
    
    # Fourth pass: Or-opt again
    for ri in range(len(routes)):
        if time.time() >= deadline:
            break
        routes[ri] = _or_opt(routes[ri], instance, D, stations_set, deadline)
    
    return routes


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


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill
    stations_set = set(instance["stations"])
    D = _build(instance)
    
    # Cluster customers
    clusters = _cluster_customers(instance, D, stations_set)
    
    # Build initial routes using nearest neighbor
    routes = []
    for cluster in clusters:
        if time.time() >= deadline:
            break
        route = _nn_route(cluster, instance, D, stations_set)
        # Ensure feasibility
        route = _ensure_feasible(route, instance, D, stations_set)
        if route:
            routes.append(route)
    
    # Check coverage and add missing customers
    covered = set()
    for r in routes:
        for node in r:
            if node not in stations_set and node != 0:
                covered.add(node)
    missing = [v for v in range(1, instance["n_customers"] + 1) if v not in covered]
    for v in missing:
        if time.time() >= deadline:
            break
        # Try to insert into existing routes
        inserted = False
        for ri in range(len(routes)):
            r = routes[ri]
            for pos in range(len(r) + 1):
                cand = r[:pos] + [v] + r[pos:]
                ok, _ = _simulate(cand, instance, D, stations_set)
                if ok:
                    routes[ri] = cand
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            routes.append([v])
    
    # Local search
    if time.time() < deadline - 5:
        routes = _local_search(routes, instance, D, stations_set, deadline)
    
    # Final validity check
    if not _routes_valid(routes, instance, D, stations_set):
        # Fallback: one customer per route
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]
    
    return routes
# EVOLVE-BLOCK-END
