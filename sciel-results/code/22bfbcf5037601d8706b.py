# EVOLVE-BLOCK-START
import math
import time
import random
import heapq

_EPS = 1e-9


def plan(instance):
    t_start = time.time()
    DEADLINE = 135.0  # safety margin inside the 150s kill

    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = n_cust + len(stations) + 1
    cust_ids = list(range(1, n_cust + 1))

    # Precompute distance matrix
    cx = [c[0] for c in coords]
    cy = [c[1] for c in coords]
    dist = [[0.0] * n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
            dist[i][j] = d
            dist[j][i] = d

    is_cust = [False] * n_nodes
    for v in cust_ids:
        is_cust[v] = True
    is_station = [False] * n_nodes
    for v in stations:
        is_station[v] = True

    def trans(u, v, t, bat):
        """Transition from u (departing) to v. Returns (new_t, new_bat, d) or None."""
        d = dist[u][v]
        if d + _EPS > bat:
            return None
        nt = t + d / speed
        if nt > horizon + _EPS:
            return None
        nb = bat - d
        if is_cust[v]:
            if nt < ready[v]:
                nt = ready[v]
            if nt > due[v] + _EPS:
                return None
            nt += service[v]
            if nt > horizon + _EPS:
                return None
        elif is_station[v]:
            if nb > battery_max + _EPS:
                nb = battery_max
            add = (battery_max - nb) * recharge_rate
            nt += add
            nb = battery_max
            if nt > horizon + _EPS:
                return None
        return nt, nb, d

    def route_distance(nodes):
        if not nodes:
            return 0.0
        total = dist[0][nodes[0]]
        for i in range(1, len(nodes)):
            total += dist[nodes[i-1]][nodes[i]]
        total += dist[nodes[-1]][0]
        return total

    def check_route(nodes):
        """Check feasibility of a route (list of node ids). Returns True if valid."""
        if not nodes:
            return True
        load = sum(demands[v] for v in nodes)
        if load > capacity + _EPS:
            return False
        t = 0.0
        bat = battery_max
        u = 0
        for v in nodes:
            r = trans(u, v, t, bat)
            if r is None:
                return False
            t, bat, _ = r
            u = v
        r = trans(u, 0, t, bat)
        if r is None:
            return False
        return True

    def try_insert_station(route, pos):
        """Try inserting a station at position pos in route. Return new route or None."""
        for s in stations:
            cand = route[:pos] + [s] + route[pos:]
            if check_route(cand):
                return cand
        return None

    def repair_route(route):
        """Try to make a route feasible by inserting stations."""
        if check_route(route):
            return route
        # Try inserting stations at various positions
        for pos in range(len(route) + 1):
            new_route = try_insert_station(route, pos)
            if new_route is not None:
                return new_route
        # Try inserting at multiple positions
        for s in stations:
            for pos in range(len(route) + 1):
                cand = route[:pos] + [s] + route[pos:]
                if check_route(cand):
                    return cand
        # Try two stations
        for s1 in stations:
            for s2 in stations:
                for pos1 in range(len(route) + 1):
                    for pos2 in range(pos1 + 1, len(route) + 1):
                        cand = route[:pos1] + [s1] + route[pos1:pos2] + [s2] + route[pos2:]
                        if check_route(cand):
                            return cand
        return None

    # --- Savings-based construction ---
    def savings_construction():
        # Compute savings: d(0,i) + d(0,j) - d(i,j)
        savings = []
        for i in range(1, n_cust + 1):
            for j in range(i + 1, n_cust + 1):
                s = dist[0][i] + dist[0][j] - dist[i][j]
                savings.append((s, i, j))
        savings.sort(reverse=True)

        # Initialize: each customer is its own route
        parent = {v: v for v in cust_ids}
        routes = {v: [v] for v in cust_ids}
        loads = {v: demands[v] for v in cust_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            # Merge rb into ra
            parent[rb] = ra
            # Merge routes
            route_a = routes[ra]
            route_b = routes[rb]
            loads[ra] += loads[rb]
            if loads[ra] > capacity + _EPS:
                return False
            # Try all 4 ways to merge
            best_route = None
            best_dist = float('inf')
            candidates = [
                route_a + route_b,
                route_a + route_b[::-1],
                route_b + route_a,
                route_b + route_a[::-1],
            ]
            for cand in candidates:
                if check_route(cand):
                    d = route_distance(cand)
                    if d < best_dist:
                        best_dist = d
                        best_route = cand
            if best_route is None:
                # Try with station insertion
                for cand in candidates:
                    repaired = repair_route(cand)
                    if repaired is not None:
                        d = route_distance(repaired)
                        if d < best_dist:
                            best_dist = d
                            best_route = repaired
            if best_route is None:
                return False
            routes[ra] = best_route
            del routes[rb]
            return True

        for s, i, j in savings:
            if time.time() > DEADLINE:
                break
            if find(i) != find(j):
                union(i, j)

        # Collect final routes
        result = []
        seen_roots = set()
        for v in cust_ids:
            r = find(v)
            if r not in seen_roots:
                seen_roots.add(r)
                result.append(routes[r])
        return result

    # --- 2-opt improvement ---
    def two_opt(route, max_iter=50):
        n = len(route)
        if n < 3:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() > DEADLINE:
                return best
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    if route_distance(cand) < route_distance(best) - 1e-7:
                        if check_route(cand):
                            best = cand
                            improved = True
                        else:
                            repaired = repair_route(cand)
                            if repaired is not None and route_distance(repaired) < route_distance(best) - 1e-7:
                                best = repaired
                                improved = True
        return best

    # --- Or-opt (relocate 1-3 consecutive nodes) ---
    def or_opt(route, max_iter=30):
        n = len(route)
        if n < 4:
            return route
        best = list(route)
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() > DEADLINE:
                return best
            for L in range(1, min(4, n)):
                for i in range(n - L + 1):
                    segment = best[i:i+L]
                    rest = best[:i] + best[i+L:]
                    for j in range(len(rest) + 1):
                        if j >= i and j < i + L:
                            continue
                        cand = rest[:j] + segment + rest[j:]
                        if len(cand) != n:
                            continue
                        if route_distance(cand) < route_distance(best) - 1e-7:
                            if check_route(cand):
                                best = cand
                                improved = True
                                break
                            else:
                                repaired = repair_route(cand)
                                if repaired is not None and route_distance(repaired) < route_distance(best) - 1e-7:
                                    best = repaired
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
        return best

    # --- Inter-route relocate ---
    def relocate_customer(routes, max_passes=4):
        for _ in range(max_passes):
            if time.time() > DEADLINE:
                break
            moved = False
            for ri, r in enumerate(routes):
                if len(r) <= 1:
                    continue
                for idx, v in enumerate(list(r)):
                    if not is_cust[v]:
                        continue
                    base = r[:idx] + r[idx+1:]
                    if not check_route(base):
                        continue
                    base_dist = route_distance(base)
                    for rj, r2 in enumerate(routes):
                        if rj == ri:
                            continue
                        for k in range(len(r2) + 1):
                            cand = r2[:k] + [v] + r2[k:]
                            if check_route(cand):
                                new_total = base_dist + route_distance(cand)
                                old_total = route_distance(r) + route_distance(r2)
                                if new_total < old_total - 1e-7:
                                    routes[ri] = base
                                    routes[rj] = cand
                                    moved = True
                                    break
                        if moved:
                            break
                if moved:
                    break
            if not moved:
                break

    # --- Inter-route swap ---
    def swap_customers(routes, max_passes=2):
        for _ in range(max_passes):
            if time.time() > DEADLINE:
                break
            moved = False
            for ri, r in enumerate(routes):
                if len(r) < 2:
                    continue
                for rj, r2 in enumerate(routes):
                    if rj <= ri or len(r2) < 2:
                        continue
                    for i, v in enumerate(r):
                        if not is_cust[v]:
                            continue
                        for j, w in enumerate(r2):
                            if not is_cust[w]:
                                continue
                            # Swap v and w
                            cand1 = r[:i] + [w] + r[i+1:]
                            cand2 = r2[:j] + [v] + r2[j+1:]
                            if check_route(cand1) and check_route(cand2):
                                new_total = route_distance(cand1) + route_distance(cand2)
                                old_total = route_distance(r) + route_distance(r2)
                                if new_total < old_total - 1e-7:
                                    routes[ri] = cand1
                                    routes[rj] = cand2
                                    moved = True
                                    break
                        if moved:
                            break
                    if moved:
                        break
                if moved:
                    break
            if not moved:
                break

    # --- Main ---
    # Initial construction via savings
    routes = savings_construction()

    # Ensure all customers are served
    served = set()
    for r in routes:
        for v in r:
            if is_cust[v]:
                served.add(v)
    for v in cust_ids:
        if v not in served:
            routes.append([v])
            served.add(v)

    # Remove empty routes
    routes = [r for r in routes if r]

    # Intra-route optimization: 2-opt + or-opt
    for i in range(len(routes)):
        if time.time() > DEADLINE:
            break
        routes[i] = two_opt(routes[i])
        routes[i] = or_opt(routes[i])

    # Inter-route optimization
    relocate_customer(routes, max_passes=6)
    swap_customers(routes, max_passes=3)

    # Second round of intra-route optimization after inter-route moves
    for i in range(len(routes)):
        if time.time() > DEADLINE:
            break
        routes[i] = two_opt(routes[i], max_iter=20)

    # Final relocate pass
    relocate_customer(routes, max_passes=3)

    # Global validity check
    def global_valid():
        seen = set()
        for r in routes:
            load = sum(demands[v] for v in r if is_cust[v])
            if load > capacity + _EPS:
                return False
            t = 0.0
            bat = battery_max
            u = 0
            for v in r:
                if is_cust[v] and v in seen:
                    return False
                if is_cust[v]:
                    seen.add(v)
                rr = trans(u, v, t, bat)
                if rr is None:
                    return False
                t, bat, _ = rr
                u = v
            rr = trans(u, 0, t, bat)
            if rr is None:
                return False
            if t > horizon + _EPS:
                return False
        if seen != set(cust_ids):
            return False
        return True

    if not global_valid():
        # Fallback: one customer per route
        fb = list(cust_ids)
        fb.sort(key=lambda v: due[v])
        return [[v] for v in fb]

    return [r for r in routes if r]

# EVOLVE-BLOCK-END
