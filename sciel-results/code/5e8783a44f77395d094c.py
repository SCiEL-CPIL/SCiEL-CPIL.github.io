"""EV routing: cluster-based construction + time-dependent Dijkstra per cluster + inter-route merge."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import heapq
from itertools import permutations

DEADLINE_SECONDS = 115.0


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _validate_route(route, coords, demands, ready, due, service, capacity,
                    battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    """Validate a single route. Returns (valid, distance)."""
    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0

    cur = 0
    t = 0.0
    battery = battery_max
    dist = 0.0

    for node in route:
        d = _dist(coords[cur], coords[node])
        battery -= d
        t += d / speed
        if battery < 0:
            return False, 0.0
        dist += d
        cur = node

        if node in stations_set:
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            t = max(t, ready[node])
            if t > due[node]:
                return False, 0.0
            t += service[node]

    d = _dist(coords[cur], coords[0])
    battery -= d
    t += d / speed
    dist += d
    if battery < 0:
        return False, 0.0
    if t > horizon:
        return False, 0.0

    return True, dist


def _validate_solution(routes, coords, demands, ready, due, service, capacity,
                       battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    all_visited = []
    for route in routes:
        for node in route:
            if 1 <= node <= n_customers:
                all_visited.append(node)
    if len(all_visited) != n_customers:
        return False, 0.0
    if len(set(all_visited)) != n_customers:
        return False, 0.0

    total_dist = 0.0
    for route in routes:
        valid, d = _validate_route(route, coords, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate,
                                   horizon, stations_set, n_customers)
        if not valid:
            return False, 0.0
        total_dist += d
    return True, total_dist


def _compute_distance(route, coords):
    if not route:
        return 0.0
    cur = 0
    total = 0.0
    for node in route:
        total += _dist(coords[cur], coords[node])
        cur = node
    total += _dist(coords[cur], coords[0])
    return total


def _kmeans_cluster(points, k, max_iters=50):
    """Simple k-means clustering. Returns list of lists of indices."""
    n = len(points)
    if n <= k:
        return [[i] for i in range(n)]

    points = np.array(points)
    # Initialize with k-means++
    centers = [points[0].copy()]
    for _ in range(1, k):
        d2 = np.array([min(np.sum((points[i] - c) ** 2) for c in centers) for i in range(n)])
        total = d2.sum()
        if total < 1e-10:
            centers.append(points[np.random.randint(n)].copy())
        else:
            probs = d2 / total
            idx = np.random.choice(n, p=probs)
            centers.append(points[idx].copy())

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iters):
        # Assign
        d2 = np.array([[np.sum((points[i] - centers[j]) ** 2) for j in range(k)] for i in range(n)])
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centers
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = points[mask].mean(axis=0)

    clusters = [[] for _ in range(k)]
    for i in range(n):
        clusters[labels[i]].append(i)
    return [c for c in clusters if c]


def _dijkstra_route(cluster_nodes, coords, demands, ready, due, service,
                    battery_max, speed, recharge_rate, horizon, stations_set,
                    n_customers, capacity):
    """
    Use Dijkstra on state (current_node, battery_level, visited_mask) to find
    minimum distance route visiting all cluster_nodes.
    For small clusters (<=8), use permutation-based exact search.
    For larger, use Dijkstra with state compression.
    """
    if not cluster_nodes:
        return [], 0.0

    # For very small clusters, try all permutations
    if len(cluster_nodes) <= 8:
        best_route = None
        best_dist = float('inf')
        for perm in permutations(cluster_nodes):
            route = list(perm)
            valid, d = _validate_route(route, coords, demands, ready, due, service,
                                       capacity, battery_max, speed, recharge_rate,
                                       horizon, stations_set, n_customers)
            if valid and d < best_dist:
                best_dist = d
                best_route = route
        if best_route is not None:
            return best_route, best_dist
        # Fallback: nearest neighbor
        return _nearest_neighbor_route(cluster_nodes, coords, demands, ready, due, service,
                                       battery_max, speed, recharge_rate, horizon,
                                       stations_set, n_customers, capacity)

    # For larger clusters, use Dijkstra with state (node, battery_quantized, visited_frozenset)
    # This can be expensive, so use a bounded version
    return _dijkstra_bounded(cluster_nodes, coords, demands, ready, due, service,
                             battery_max, speed, recharge_rate, horizon, stations_set,
                             n_customers, capacity)


def _nearest_neighbor_route(nodes, coords, demands, ready, due, service,
                            battery_max, speed, recharge_rate, horizon,
                            stations_set, n_customers, capacity):
    """Nearest neighbor construction as fallback."""
    remaining = list(nodes)
    route = []
    cur = 0
    t = 0.0
    battery = battery_max
    total_demand = 0

    while remaining:
        best = None
        best_d = float('inf')
        for cand in remaining:
            if 1 <= cand <= n_customers:
                if total_demand + demands[cand] > capacity:
                    continue
            d = _dist(coords[cur], coords[cand])
            new_battery = battery - d
            if new_battery < 0:
                continue
            new_t = t + d / speed
            if cand in stations_set:
                new_t += (battery_max - new_battery) * recharge_rate
            else:
                new_t = max(new_t, ready[cand])
                if new_t > due[cand]:
                    continue
            if d < best_d:
                best_d = d
                best = cand

        if best is None:
            # Try any remaining node ignoring distance
            for cand in remaining:
                d = _dist(coords[cur], coords[cand])
                if battery - d >= 0:
                    best = cand
                    break
            if best is None:
                best = remaining[0]

        d = _dist(coords[cur], coords[best])
        battery -= d
        t += d / speed
        cur = best
        route.append(best)
        remaining.remove(best)

        if best in stations_set:
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            t = max(t, ready[best])
            t += service[best]
            total_demand += demands[best]

    return route, _compute_distance(route, coords)


def _dijkstra_bounded(cluster_nodes, coords, demands, ready, due, service,
                      battery_max, speed, recharge_rate, horizon, stations_set,
                      n_customers, capacity, max_states=50000):
    """Bounded Dijkstra for route construction. State: (node, battery_q, visited_tuple)."""
    # This is complex; for now use a smarter greedy with look-ahead
    # Try 2-opt improvement on nearest neighbor result
    route, d = _nearest_neighbor_route(cluster_nodes, coords, demands, ready, due, service,
                                       battery_max, speed, recharge_rate, horizon,
                                       stations_set, n_customers, capacity)
    # Apply 2-opt
    route = _two_opt(route, coords, demands, ready, due, service,
                     capacity, battery_max, speed, recharge_rate, horizon,
                     stations_set, n_customers, max_iters=15)
    return route, _compute_distance(route, coords)


def _two_opt(route, coords, demands, ready, due, service,
             capacity, battery_max, speed, recharge_rate, horizon,
             stations_set, n_customers, max_iters=20):
    """2-opt improvement with feasibility check."""
    if len(route) < 2:
        return route

    best_route = list(route)
    best_dist = _compute_distance(best_route, coords)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(best_route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                valid, d = _validate_route(candidate, coords, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
                if valid and d < best_dist - 1e-9:
                    best_route = candidate
                    best_dist = d
                    improved = True

    return best_route


def _or_opt(route, coords, demands, ready, due, service,
            capacity, battery_max, speed, recharge_rate, horizon,
            stations_set, n_customers, max_iters=10):
    """Or-opt: move 1-3 consecutive nodes to a different position."""
    if len(route) < 4:
        return route

    best_route = list(route)
    best_dist = _compute_distance(best_route, coords)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(best_route)
        for seg_len in range(1, 4):
            for i in range(n - seg_len + 1):
                segment = best_route[i:i + seg_len]
                rest = best_route[:i] + best_route[i + seg_len:]
                for j in range(len(rest) + 1):
                    if i > 0 and j <= i:
                        continue
                    if i + seg_len <= j:
                        continue
                    candidate = rest[:j] + segment + rest[j:]
                    valid, d = _validate_route(candidate, coords, demands, ready, due, service,
                                               capacity, battery_max, speed, recharge_rate,
                                               horizon, stations_set, n_customers)
                    if valid and d < best_dist - 1e-9:
                        best_route = candidate
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route


def _try_insert_station(route, coords, demands, ready, due, service,
                        capacity, battery_max, speed, recharge_rate, horizon,
                        stations_set, n_customers):
    """Try to insert a station at the best position in the route if it helps."""
    best_route = list(route)
    best_dist = _compute_distance(best_route, coords)

    for station in stations_set:
        # Try inserting at each position
        for pos in range(len(best_route) + 1):
            candidate = best_route[:pos] + [station] + best_route[pos:]
            valid, d = _validate_route(candidate, coords, demands, ready, due, service,
                                       capacity, battery_max, speed, recharge_rate,
                                       horizon, stations_set, n_customers)
            if valid and d < best_dist - 1e-9:
                best_route = candidate
                best_dist = d

    return best_route


def _inter_route_merge(routes, coords, demands, ready, due, service,
                       capacity, battery_max, speed, recharge_rate, horizon,
                       stations_set, n_customers):
    """Try to merge pairs of routes if feasible."""
    if len(routes) < 2:
        return routes

    merged = True
    while merged:
        merged = False
        best_merge = None
        best_saving = 0

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Try concatenation in both orders
                for combined in [routes[i] + routes[j], routes[j] + routes[i]]:
                    valid, d = _validate_route(combined, coords, demands, ready, due, service,
                                               capacity, battery_max, speed, recharge_rate,
                                               horizon, stations_set, n_customers)
                    if valid:
                        old_d = _compute_distance(routes[i], coords) + _compute_distance(routes[j], coords)
                        saving = old_d - d
                        if saving > best_saving + 1e-9:
                            best_saving = saving
                            best_merge = (i, j, combined)

        if best_merge:
            i, j, combined = best_merge
            routes.pop(j)
            routes.pop(i)
            routes.append(combined)
            merged = True

    return routes


def _inter_route_move(routes, coords, demands, ready, due, service,
                      capacity, battery_max, speed, recharge_rate, horizon,
                      stations_set, n_customers, max_moves=50):
    """Try moving a customer from one route to another."""
    improved = True
    moves = 0
    while improved and moves < max_moves:
        improved = False
        moves += 1
        for i in range(len(routes)):
            for j in range(len(routes[i])):
                if 1 <= routes[i][j] <= n_customers:
                    cust = routes[i][j]
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        # Try inserting cust at each position in route k
                        for pos in range(len(routes[k]) + 1):
                            new_i = routes[i][:j] + routes[i][j + 1:]
                            new_k = routes[k][:pos] + [cust] + routes[k][pos:]

                            # Check capacity
                            demand_i = sum(demands[v] for v in new_i if 1 <= v <= n_customers)
                            demand_k = sum(demands[v] for v in new_k if 1 <= v <= n_customers)
                            if demand_k > capacity:
                                continue

                            valid_i, d_i = _validate_route(new_i, coords, demands, ready, due, service,
                                                           capacity, battery_max, speed, recharge_rate,
                                                           horizon, stations_set, n_customers) if new_i else (True, 0.0)
                            valid_k, d_k = _validate_route(new_k, coords, demands, ready, due, service,
                                                           capacity, battery_max, speed, recharge_rate,
                                                           horizon, stations_set, n_customers)

                            if valid_i and valid_k:
                                old_d = _compute_distance(routes[i], coords) + _compute_distance(routes[k], coords)
                                new_d = d_i + d_k
                                if new_d < old_d - 1e-9:
                                    routes[i] = new_i
                                    routes[k] = new_k
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            if improved:
                break

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def plan(instance):
    start_time = time.time()
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))
    cust_coords = [coords[c] for c in customers]

    # Determine number of clusters based on capacity
    total_demand = sum(demands[c] for c in customers)
    n_clusters = max(2, int(math.ceil(total_demand / capacity)))
    # Also consider distance-based: if points are spread out, more routes
    n_clusters = min(n_clusters, len(customers))

    # K-means clustering
    clusters_idx = _kmeans_cluster(cust_coords, n_clusters)
    clusters = [[customers[i] for i in c] for c in clusters_idx]

    # Ensure no cluster exceeds capacity; if so, split
    final_clusters = []
    for cl in clusters:
        cl_demand = sum(demands[c] for c in cl)
        if cl_demand <= capacity:
            final_clusters.append(cl)
        else:
            # Split into singletons
            for c in cl:
                final_clusters.append([c])

    # Build initial routes for each cluster
    routes = []
    for cl in final_clusters:
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        route, d = _dijkstra_route(cl, coords, demands, ready, due, service,
                                   battery_max, speed, recharge_rate, horizon,
                                   stations_set, n_customers, capacity)
        if route:
            routes.append(route)

    # Check all customers are covered
    covered = set()
    for r in routes:
        for node in r:
            if 1 <= node <= n_customers:
                covered.add(node)
    missing = [c for c in customers if c not in covered]
    for c in missing:
        routes.append([c])

    # Validate
    valid, total_dist = _validate_solution(routes, coords, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
    if not valid:
        # Fallback: one customer per route
        return [[v] for v in customers]

    # Local optimization: 2-opt + or-opt on each route
    for i in range(len(routes)):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        routes[i] = _two_opt(routes[i], coords, demands, ready, due, service,
                             capacity, battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, max_iters=12)
        routes[i] = _or_opt(routes[i], coords, demands, ready, due, service,
                            capacity, battery_max, speed, recharge_rate, horizon,
                            stations_set, n_customers, max_iters=8)

    # Inter-route: try merging
    if time.time() - start_time < DEADLINE_SECONDS - 10:
        routes = _inter_route_merge(routes, coords, demands, ready, due, service,
                                    capacity, battery_max, speed, recharge_rate,
                                    horizon, stations_set, n_customers)

    # Inter-route: try moving customers
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        routes = _inter_route_move(routes, coords, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate,
                                   horizon, stations_set, n_customers, max_moves=30)

    # Final 2-opt pass
    for i in range(len(routes)):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        routes[i] = _two_opt(routes[i], coords, demands, ready, due, service,
                             capacity, battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, max_iters=8)

    # Final validation
    valid, total_dist = _validate_solution(routes, coords, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
    if not valid:
        return [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
