"""Rewritten: state-space shortest path + clustering + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq


def plan(instance):
    start = time.time()
    deadline = start + 140.0

    coords = np.asarray(instance["coords"], dtype=float)
    demands = np.asarray(instance["demands"], dtype=float)
    ready = np.asarray(instance["ready"], dtype=float)
    due = np.asarray(instance["due"], dtype=float)
    service = np.asarray(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    dist = np.sqrt(d2)

    customers = list(range(1, n_customers + 1))

    # ---------- Simulation with charging ----------
    def simulate_with_charge(route, allow_charge=True):
        """Simulate route, auto-inserting charges at stations if needed.
        Returns (ok, total_dist, end_time, final_route_with_stations)."""
        if not route:
            return True, 0.0, 0.0, []
        load = float(demands[route].sum())
        if load > capacity + 1e-9:
            return False, 0.0, 0.0, []

        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        final_route = []

        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                # Need to charge: find best station to visit first
                if not allow_charge:
                    return False, 0.0, 0.0, []
                # Find station that minimizes extra distance while providing enough battery
                best_station = None
                best_extra = float('inf')
                for st in stations:
                    d_to_st = dist[prev, st]
                    d_st_to_v = dist[st, v]
                    extra = d_to_st + d_st_to_v - d
                    if extra < best_extra:
                        # Check: after charging at st, can we reach v?
                        # Battery at st before charge: battery - d_to_st
                        # After charge: battery_cap
                        # Need: battery_cap >= d_st_to_v
                        if battery_cap >= d_st_to_v + 1e-9:
                            best_extra = extra
                            best_station = st
                if best_station is None:
                    return False, 0.0, 0.0, []
                st = best_station
                d_to_st = dist[prev, st]
                battery -= d_to_st
                t += d_to_st / speed
                total += d_to_st
                # Charge at station
                charge_time = (battery_cap - battery) * recharge_rate
                t += charge_time
                battery = battery_cap
                prev = st
                final_route.append(st)
                # Now go to v
                d_st_to_v = dist[st, v]
                battery -= d_st_to_v
                t += d_st_to_v / speed
                total += d_st_to_v
                prev = v
            else:
                battery -= d
                t += d / speed
                total += d
                prev = v

            # Time window check
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, []
            t += service[v]
            final_route.append(v)

        # Return to depot
        d = dist[prev, 0]
        if d > battery + 1e-9:
            if allow_charge:
                # Need to charge before returning
                best_station = None
                best_extra = float('inf')
                for st in stations:
                    d_to_st = dist[prev, st]
                    d_st_to_0 = dist[st, 0]
                    extra = d_to_st + d_st_to_0 - d
                    if extra < best_extra and battery_cap >= d_st_to_0 + 1e-9:
                        best_extra = extra
                        best_station = st
                if best_station is None:
                    return False, 0.0, 0.0, []
                st = best_station
                d_to_st = dist[prev, st]
                battery -= d_to_st
                t += d_to_st / speed
                total += d_to_st
                charge_time = (battery_cap - battery) * recharge_rate
                t += charge_time
                battery = battery_cap
                prev = st
                final_route.append(st)
                d_st_to_0 = dist[st, 0]
                battery -= d_st_to_0
                t += d_st_to_0 / speed
                total += d_st_to_0
                prev = 0
            else:
                return False, 0.0, 0.0, []
        else:
            battery -= d
            t += d / speed
            total += d
            prev = 0

        if t > horizon + 1e-9:
            return False, 0.0, 0.0, []
        return True, total, t, final_route

    def simulate(route):
        """Quick feasibility + distance (no station insertion, just check)."""
        if not route:
            return True, 0.0, 0.0
        load = float(demands[route].sum())
        if load > capacity + 1e-9:
            return False, 0.0, 0.0
        t = 0.0
        battery = battery_cap
        total = 0.0
        prev = 0
        for v in route:
            d = dist[prev, v]
            if d > battery + 1e-9:
                return False, 0.0, 0.0
            battery -= d
            t += d / speed
            total += d
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0
            t += service[v]
            prev = v
        d = dist[prev, 0]
        if d > battery + 1e-9:
            return False, 0.0, 0.0
        t += d / speed
        if t > horizon + 1e-9:
            return False, 0.0, 0.0
        return True, total, t

    def route_dist(route):
        if not route:
            return 0.0
        t = dist[0, route[0]]
        for i in range(len(route) - 1):
            t += dist[route[i], route[i + 1]]
        t += dist[route[-1], 0]
        return t

    # ---------- State-space shortest path for a cluster ----------
    # State: (visited_mask, current_node, battery_quantized, time_quantized)
    # For small clusters (<=8), use bitmask DP; for larger, use greedy + local search
    def tsp_dp(cluster):
        """Solve TSP with time windows and battery for a small cluster using DP.
        Returns (ok, best_route, best_dist)."""
        m = len(cluster)
        if m == 0:
            return True, [], 0.0
        if m == 1:
            v = cluster[0]
            ok, d, t = simulate([v])
            if ok:
                return True, [v], d
            return False, [], float('inf')

        # Use bitmask DP for m <= 10
        if m > 10:
            return None  # Too large for DP

        # Map cluster indices
        # State: (mask, last_idx) -> (min_time, min_dist, battery_at_end)
        # Actually we need to track battery and time, which makes it complex.
        # Simplified: track (time, dist) and assume battery is feasible if we check at end.
        # Better: for each state, track the minimum time to reach with a given battery.
        # This is getting complex. Let's use a simpler approach:
        # For m <= 8, enumerate all permutations is 40320, feasible.
        if m <= 8:
            best = None
            best_dist = float('inf')
            from itertools import permutations
            for perm in permutations(cluster):
                ok, d, t = simulate(list(perm))
                if ok and d < best_dist:
                    best_dist = d
                    best = list(perm)
            if best is not None:
                return True, best, best_dist
            return False, [], float('inf')

        # For m = 9-10, use greedy + 2-opt
        return None

    def greedy_tsp(cluster):
        """Nearest neighbor with time window and battery feasibility."""
        if not cluster:
            return []
        remaining = list(cluster)
        route = []
        prev = 0
        t = 0.0
        battery = battery_cap

        while remaining:
            best_v = None
            best_d = float('inf')
            for v in remaining:
                d = dist[prev, v]
                if d > battery + 1e-9:
                    continue
                new_t = t + d / speed
                if new_t < ready[v]:
                    new_t = ready[v]
                if new_t > due[v] + 1e-9:
                    continue
                if d < best_d:
                    best_d = d
                    best_v = v
            if best_v is None:
                # Try any feasible v (relax distance)
                for v in remaining:
                    d = dist[prev, v]
                    if d > battery + 1e-9:
                        continue
                    new_t = t + d / speed
                    if new_t < ready[v]:
                        new_t = ready[v]
                    if new_t > due[v] + 1e-9:
                        continue
                    best_v = v
                    break
            if best_v is None:
                return None  # Infeasible
            remaining.remove(best_v)
            d = dist[prev, best_v]
            battery -= d
            t += d / speed
            if t < ready[best_v]:
                t = ready[best_v]
            t += service[best_v]
            route.append(best_v)
            prev = best_v

        # Check return
        d = dist[prev, 0]
        if d > battery + 1e-9:
            return None
        t += d / speed
        if t > horizon + 1e-9:
            return None
        return route

    def two_opt(route):
        if len(route) < 3:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < 5000:
            improved = False
            it += 1
            L = len(best)
            for i in range(L - 1):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                for j in range(i + 1, L):
                    c = best[j]
                    d = best[j + 1] if j + 1 < L else 0
                    old = dist[a, b] + dist[c, d]
                    new = dist[a, c] + dist[b, d]
                    if new < old - 1e-9:
                        cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                        ok, tot, _ = simulate(cand)
                        if ok:
                            best = cand
                            improved = True
                            break
                if improved:
                    break
        return best

    def or_opt(route):
        if len(route) < 3:
            return route
        best = list(route)
        best_dist = route_dist(best)
        improved = True
        it = 0
        while improved and it < 2000:
            improved = False
            it += 1
            L = len(best)
            for seg_len in (1, 2, 3):
                for i in range(L - seg_len + 1):
                    seg = best[i:i + seg_len]
                    rest = best[:i] + best[i + seg_len:]
                    if not rest:
                        continue
                    for pos in range(len(rest) + 1):
                        cand = rest[:pos] + seg + rest[pos:]
                        if cand == best:
                            continue
                        ok, tot, _ = simulate(cand)
                        if ok and tot < best_dist - 1e-9:
                            best = cand
                            best_dist = tot
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best

    # ---------- Clustering ----------
    def cluster_customers(method='spatial', k=None, seed=0):
        """Cluster customers into groups for routing."""
        rng = random.Random(seed)
        if method == 'spatial':
            # Simple k-means-like clustering with k based on capacity
            # Estimate number of routes needed
            total_demand = float(demands[customers].sum())
            est_routes = max(1, int(math.ceil(total_demand / capacity)))
            # Also consider time windows: customers with far-apart windows need separate routes
            k = min(est_routes + 2, len(customers))

            # Initialize centers randomly
            indices = list(range(len(customers)))
            centers_idx = rng.sample(indices, k)
            centers = [coords[customers[i]] for i in centers_idx]

            for _ in range(20):
                # Assign
                assign = [0] * len(customers)
                for i, v in enumerate(customers):
                    best_c = 0
                    best_d = float('inf')
                    for c in range(k):
                        d = np.linalg.norm(coords[v] - centers[c])
                        if d < best_d:
                            best_d = d
                            best_c = c
                    assign[i] = best_c

                # Update centers
                new_centers = [None] * k
                for c in range(k):
                    members = [customers[i] for i in range(len(customers)) if assign[i] == c]
                    if members:
                        new_centers[c] = coords[members].mean(axis=0)
                    else:
                        new_centers[c] = centers[c]

                # Check convergence
                if all(np.allclose(centers[c], new_centers[c]) for c in range(k)):
                    break
                centers = new_centers

            clusters = [[] for _ in range(k)]
            for i, v in enumerate(customers):
                clusters[assign[i]].append(v)
            # Remove empty clusters
            clusters = [c for c in clusters if c]
            return clusters

        elif method == 'time_window':
            # Cluster by time window compatibility
            # Two customers are compatible if their windows overlap or are close
            # Use a greedy approach
            clusters = []
            unassigned = list(customers)
            while unassigned:
                if time.time() > deadline - 20:
                    break
                # Pick the customer with the earliest due date
                v0 = min(unassigned, key=lambda v: due[v])
                cluster = [v0]
                unassigned.remove(v0)
                # Add compatible customers
                changed = True
                while changed:
                    changed = False
                    for v in list(unassigned):
                        # Check if v is compatible with all in cluster
                        compatible = True
                        for u in cluster:
                            # Check if windows are close enough
                            if due[u] < ready[v] - 50 or due[v] < ready[u] - 50:
                                compatible = False
                                break
                        if compatible:
                            # Check capacity
                            load = float(demands[cluster].sum()) + demands[v]
                            if load <= capacity + 1e-9:
                                cluster.append(v)
                                unassigned.remove(v)
                                changed = True
                clusters.append(cluster)
            return clusters

        else:  # 'random'
            k = max(1, int(math.ceil(float(demands[customers].sum()) / capacity)))
            k = min(k + 1, len(customers))
            clusters = [[] for _ in range(k)]
            rng.shuffle(customers)
            for i, v in enumerate(customers):
                clusters[i % k].append(v)
            return [c for c in clusters if c]

    # ---------- Build routes from clusters ----------
    def build_routes_from_clusters(clusters, seed=0):
        """For each cluster, find a feasible route."""
        routes = []
        for cluster in clusters:
            if not cluster:
                continue
            # Try DP for small clusters
            if len(cluster) <= 8:
                result = tsp_dp(cluster)
                if result is not None:
                    ok, route, d = result
                    if ok:
                        routes.append(route)
                        continue
            # Try greedy
            route = greedy_tsp(cluster)
            if route is None:
                # Split cluster: try each customer individually
                for v in cluster:
                    routes.append([v])
                continue
            # Improve with local search
            route = two_opt(route)
            route = or_opt(route)
            routes.append(route)
        return routes

    # ---------- Inter-route optimization ----------
    def relocate_opt(routes, max_iter=300):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                if time.time() > deadline - 15:
                    return routes
                for r2 in range(n_routes):
                    if r1 == r2:
                        continue
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(len(route1)):
                        v = route1[i]
                        load2 = float(demands[route2].sum()) + demands[v]
                        if load2 > capacity + 1e-9:
                            continue
                        new_route1 = route1[:i] + route1[i+1:]
                        best_pos = -1
                        best_cost = float('inf')
                        L2 = len(route2)
                        for pos in range(L2 + 1):
                            cand = route2[:pos] + [v] + route2[pos:]
                            ok, tot, _ = simulate(cand)
                            if ok and tot < best_cost:
                                best_cost = tot
                                best_pos = pos
                        if best_pos >= 0:
                            new_route2 = route2[:best_pos] + [v] + route2[best_pos:]
                            ok1, d1, _ = simulate(new_route1)
                            if ok1:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = d1 + best_cost
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def swap_opt(routes, max_iter=150):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n_routes = len(routes)
            for r1 in range(n_routes):
                if time.time() > deadline - 15:
                    return routes
                for r2 in range(r1 + 1, n_routes):
                    route1 = routes[r1]
                    route2 = routes[r2]
                    for i in range(len(route1)):
                        v1 = route1[i]
                        for j in range(len(route2)):
                            v2 = route2[j]
                            load1 = float(demands[route1].sum()) - demands[v1] + demands[v2]
                            load2 = float(demands[route2].sum()) - demands[v2] + demands[v1]
                            if load1 > capacity + 1e-9 or load2 > capacity + 1e-9:
                                continue
                            new_route1 = route1[:i] + [v2] + route1[i+1:]
                            new_route2 = route2[:j] + [v1] + route2[j+1:]
                            ok1, d1, _ = simulate(new_route1)
                            ok2, d2, _ = simulate(new_route2)
                            if ok1 and ok2:
                                old_total = route_dist(route1) + route_dist(route2)
                                new_total = d1 + d2
                                if new_total < old_total - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # ---------- Main: multiple strategies ----------
    best_routes = None
    best_score = float('inf')

    strategies = [
        ('spatial', 0),
        ('spatial', 1),
        ('spatial', 2),
        ('time_window', 0),
        ('time_window', 1),
        ('random', 0),
        ('random', 1),
        ('random', 2),
        ('spatial', 3),
        ('spatial', 4),
    ]

    for strat, seed in strategies:
        if time.time() > deadline - 25:
            break
        clusters = cluster_customers(strat, seed=seed)
        routes = build_routes_from_clusters(clusters, seed=seed)

        # Local search
        for r in range(len(routes)):
            if time.time() > deadline - 10:
                break
            routes[r] = two_opt(routes[r])
            routes[r] = or_opt(routes[r])

        # Inter-route
        if time.time() < deadline - 20:
            routes = relocate_opt(routes)
        if time.time() < deadline - 15:
            routes = swap_opt(routes)

        # Second pass
        for r in range(len(routes)):
            if time.time() > deadline - 5:
                break
            routes[r] = two_opt(routes[r])
            routes[r] = or_opt(routes[r])

        score = sum(route_dist(r) for r in routes)
        if score < best_score:
            best_score = score
            best_routes = [list(r) for r in routes]

    # Fallback
    if best_routes is None:
        best_routes = [[v] for v in customers]
    else:
        served = set()
        for r in best_routes:
            for v in r:
                served.add(v)
        missing = [v for v in customers if v not in served]
        for v in missing:
            best_routes.append([v])

    return best_routes
# EVOLVE-BLOCK-END
