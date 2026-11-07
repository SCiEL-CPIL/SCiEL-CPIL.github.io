# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
import heapq


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin vs 150s wall-clock kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = int(instance["n_customers"])
    stations = list(instance["stations"])
    capacity = float(instance["capacity"])
    battery_cap = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])

    n_nodes = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(-1))

    customers = list(range(1, n_customers + 1))
    cust_set = set(customers)
    station_set = set(stations)
    is_customer = np.zeros(n_nodes, dtype=bool)
    is_station = np.zeros(n_nodes, dtype=bool)
    for c in customers:
        is_customer[c] = True
    for s in stations:
        is_station[s] = True

    cust_demand = np.zeros(n_nodes)
    for c in customers:
        cust_demand[c] = demands[c]

    def simulate_fast(route):
        if not route:
            return (0.0, 0.0, battery_cap)
        t = 0.0
        bat = battery_cap
        cd = 0.0
        prev = 0
        for node in route:
            d = dist[prev, node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return None
            cd += d
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            prev = node
        d = dist[prev, 0]
        t += d / speed
        bat -= d
        cd += d
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
        return (cd, t, bat)

    def route_demand(route):
        s = 0.0
        for v in route:
            if is_customer[v]:
                s += cust_demand[v]
        return s

    def total_dist(routes):
        tot = 0.0
        for r in routes:
            s = simulate_fast(r)
            if s is None:
                return float("inf")
            tot += s[0]
        return tot

    def best_station_fix(route):
        """Insert cheapest single station if needed to make route feasible."""
        s = simulate_fast(route)
        if s is not None:
            return list(route), s[0]
        t = 0.0
        bat = battery_cap
        cur = 0
        neg_pos = -1
        for idx, node in enumerate(route):
            d = dist[cur, node]
            t += d / speed
            bat -= d
            if bat < -1e-9 and neg_pos == -1:
                neg_pos = idx
            if is_customer[node]:
                if t < ready[node]:
                    t = ready[node]
                t += service[node]
            elif is_station[node]:
                t += (battery_cap - bat) * recharge_rate
                bat = battery_cap
            cur = node
        if neg_pos == -1:
            d = dist[cur, 0]
            bat -= d
            if bat < -1e-9:
                neg_pos = len(route)
            else:
                return None, None

        best_route = None
        best_d = float('inf')
        search_start = max(0, neg_pos - 2)
        search_end = min(len(route) + 1, neg_pos + 3)
        for p in range(search_start, search_end + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        for p in range(len(route) + 1):
            for st in stations:
                cand = list(route[:p]) + [st] + list(route[p:])
                if route_demand(cand) > capacity:
                    continue
                sc = simulate_fast(cand)
                if sc is not None:
                    if sc[0] < best_d:
                        best_route, best_d = cand, sc[0]
        if best_route is not None:
            return best_route, best_d
        return None, None

    def greedy_construct(order):
        routes = []
        for c in order:
            placed = False
            best = None
            best_cd = float('inf')
            for ri, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                positions = []
                for pos in range(len(r) + 1):
                    prev = r[pos - 1] if pos > 0 else 0
                    nxt = r[pos] if pos < len(r) else 0
                    delta = dist[prev, c] + dist[c, nxt] - dist[prev, nxt]
                    positions.append((delta, pos))
                positions.sort()
                
                for delta, pos in positions:
                    cand = list(r[:pos]) + [c] + list(r[pos:])
                    fixed, cd = best_station_fix(cand)
                    if fixed is not None:
                        if cd < best_cd:
                            best = (ri, fixed, cd)
                            best_cd = cd
                        break
            if best is not None:
                routes[best[0]] = best[1]
                placed = True
            if not placed:
                fixed, cd = best_station_fix([c])
                if fixed is None:
                    fixed = [c]
                routes.append(fixed)
        return routes

    def two_opt(route, max_iter=200, tol=1e-7):
        if len(route) < 2:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < n else 0
                    old_len = dist[a, b] + dist[c, d]
                    new_len = dist[a, c] + dist[b, d]
                    if new_len >= old_len - tol:
                        continue
                    nr = list(route[:i]) + list(reversed(route[i:j + 1])) + list(route[j + 1:])
                    if route_demand(nr) > capacity:
                        continue
                    sn = simulate_fast(nr)
                    if sn is None:
                        continue
                    if sn[0] < cur_d - tol:
                        route = nr
                        cur_d = sn[0]
                        improved = True
            if time.time() > deadline - 5.0:
                break
        return route

    def or_opt(route, max_iter=50):
        if len(route) < 3:
            return route
        cur = simulate_fast(route)
        if cur is None:
            return route
        cur_d = cur[0]
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i + length]
                    remaining = list(route[:i]) + list(route[i + length:])
                    if not remaining:
                        continue
                    for j in range(len(remaining) + 1):
                        new_route = list(remaining[:j]) + segment + list(remaining[j:])
                        if new_route == route:
                            continue
                        if route_demand(new_route) > capacity:
                            continue
                        sn = simulate_fast(new_route)
                        if sn is None:
                            continue
                        if sn[0] < cur_d - 1e-6:
                            route = new_route
                            cur_d = sn[0]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if time.time() > deadline - 5.0:
                break
        return route

    def cross_route_or_opt(routes, max_iters=20):
        """Move segments (length 1-3) between routes to reduce total distance."""
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            for i in range(len(routes)):
                ri = routes[i]
                if len(ri) < 2:
                    continue
                for length in [1, 2, 3]:
                    if len(ri) < length + 1:
                        continue
                    for seg_start in range(len(ri) - length + 1):
                        segment = ri[seg_start:seg_start + length]
                        if not any(is_customer[c] for c in segment):
                            continue
                        remaining_i = list(ri[:seg_start]) + list(ri[seg_start + length:])
                        if not remaining_i:
                            continue
                        for j in range(len(routes)):
                            if i == j:
                                continue
                            if route_demand(routes[j]) + sum(demands[c] for c in segment if is_customer[c]) > capacity:
                                continue
                            rj = routes[j]
                            best_cost = float('inf')
                            best_new_i = None
                            best_new_j = None
                            for pos in range(len(rj) + 1):
                                cand_j = list(rj[:pos]) + segment + list(rj[pos:])
                                fixed_j, cd_j = best_station_fix(cand_j)
                                if fixed_j is None:
                                    continue
                                fixed_i, cd_i = best_station_fix(remaining_i) if remaining_i else (remaining_i, 0.0)
                                if fixed_i is None:
                                    continue
                                total_cd = cd_i + cd_j
                                if total_cd < best_cost:
                                    best_cost = total_cd
                                    best_new_i = fixed_i
                                    best_new_j = fixed_j
                            if best_new_j is not None:
                                old_total = simulate_fast(ri)[0] + simulate_fast(rj)[0]
                                if best_cost < old_total - 1e-6:
                                    routes[i] = best_new_i
                                    routes[j] = best_new_j
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    def local_search(routes, max_iters=50):
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False

            for i in range(len(routes)):
                new_route = two_opt(routes[i], 30)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            for i in range(len(routes)):
                new_route = or_opt(routes[i], 20)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # Relocate moves between routes
            for i in range(len(routes)):
                for pos_i, c in enumerate(routes[i]):
                    if not is_customer[c]:
                        continue
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        if route_demand(routes[j]) + demands[c] > capacity:
                            continue
                        new_route_i = list(routes[i][:pos_i]) + list(routes[i][pos_i + 1:])
                        best_pos = -1
                        best_cost = float('inf')
                        best_new_j = None
                        for pos in range(len(routes[j]) + 1):
                            cand = list(routes[j][:pos]) + [c] + list(routes[j][pos:])
                            fixed, cd = best_station_fix(cand)
                            if fixed is not None and cd < best_cost:
                                best_cost = cd
                                best_pos = pos
                                best_new_j = fixed
                        if best_pos >= 0:
                            sim_i = simulate_fast(new_route_i) if new_route_i else (0.0, True, 0.0, battery_cap)
                            sim_j = simulate_fast(best_new_j)
                            if sim_i is None or sim_j is None:
                                continue
                            old_dist = simulate_fast(routes[i])[0] + simulate_fast(routes[j])[0]
                            new_dist = sim_i[0] + sim_j[0]
                            if new_dist < old_dist - 1e-6:
                                routes[i] = new_route_i
                                routes[j] = best_new_j
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

            # Cross-route OR-opt
            if not improved:
                routes = cross_route_or_opt(routes, 10)
                if time.time() > deadline - 5.0:
                    break

            routes = [r for r in routes if r]
            if not improved:
                break
        return routes

    def reoptimize_subset(routes, removed_customers):
        """
        Advanced LNS step:
        1. Identify routes affected by removed customers.
        2. Remove customers from those routes.
        3. Reconstruct the affected routes using a greedy-nearest-neighbor + local search approach
           specifically for the set of (remaining + removed) customers in those routes.
        4. Re-insert any customers that couldn't be fit into the reconstructed routes into other routes.
        """
        if not removed_customers:
            return routes
        
        removed_set = set(removed_customers)
        
        # Identify affected routes and collect customers to re-optimize
        # We will re-optimize the entire content of any route that had a removed customer
        # This is a strong neighborhood search
        affected_indices = []
        for i, r in enumerate(routes):
            if any(c in removed_set for c in r if is_customer[c]):
                affected_indices.append(i)
        
        if not affected_indices:
            return routes

        # Extract customers from affected routes
        # We keep the stations? No, we strip stations too to allow re-routing.
        # We separate customers and non-customers (stations)
        # Actually, it's better to re-optimize the customers and re-insert stations via best_station_fix
        
        all_customers_to_reoptimize = []
        for idx in affected_indices:
            for c in routes[idx]:
                if is_customer[c]:
                    all_customers_to_reoptimize.append(c)
        
        # Clear the affected routes
        for idx in affected_indices:
            routes[idx] = []
        
        # Re-insert all these customers using a greedy approach into the current routes
        # This includes the empty affected routes and other routes
        # Sort by due time or distance? Let's use a greedy nearest neighbor insertion
        # To improve quality, we will try to insert them in a way that minimizes distance increase
        
        # Simple strategy: Sort by due date, then insert greedily
        sorted_to_insert = sorted(all_customers_to_reoptimize, key=lambda c: due[c])
        
        for c in sorted_to_insert:
            best_cost = float('inf')
            best_route_idx = -1
            best_new_route = None
            
            for i, r in enumerate(routes):
                if route_demand(r) + demands[c] > capacity:
                    continue
                
                # Find best position
                for pos in range(len(r) + 1):
                    cand = list(r[:pos]) + [c] + list(r[pos:])
                    fixed, cd = best_station_fix(cand)
                    if fixed is not None and cd < best_cost:
                        best_cost = cd
                        best_route_idx = i
                        best_new_route = fixed
            
            if best_route_idx >= 0:
                routes[best_route_idx] = best_new_route
            else:
                # If no place found, create a new route
                fixed, cd = best_station_fix([c])
                if fixed is None:
                    fixed = [c]
                routes.append(fixed)
        
        # Remove empty routes
        routes = [r for r in routes if r]
        
        # Run local search on the modified routes to polish
        # Only search the affected ones + neighbors? Just do a full local search pass
        # But limit iterations to save time
        routes = local_search(routes, 10)
        
        return routes

    # Initial construction sorted by due time
    sorted_customers = sorted(customers, key=lambda v: due[v])
    routes = greedy_construct(sorted_customers)
    routes = local_search(routes, 40)
    best_routes = [list(r) for r in routes]
    best_dist = total_dist(best_routes)

    # LNS iterations
    rng = random.Random(42)
    lns_iter = 0
    while lns_iter < 500:
        if time.time() > deadline - 10.0:
            break
        lns_iter += 1

        n_remove = rng.randint(3, min(8, len(customers)))
        removed = rng.sample(customers, n_remove)

        routes_copy = [list(r) for r in best_routes]
        routes_copy = reoptimize_subset(routes_copy, removed)

        d = total_dist(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d

    # Try different initial constructions
    if time.time() < deadline - 10.0:
        for _ in range(5):
            random_order = customers[:]
            rng.shuffle(random_order)
            built_routes = greedy_construct(random_order)
            built_routes = local_search(built_routes, 30)
            d2 = total_dist(built_routes)
            if d2 < best_dist - 1e-6:
                best_routes = [list(r) for r in built_routes]
                best_dist = d2

    # Final polish - high precision pass
    if time.time() < deadline - 5.0:
        for i in range(len(best_routes)):
            best_routes[i] = two_opt(best_routes[i], 200)
            best_routes[i] = or_opt(best_routes[i], 50)
        # One more round of cross-route moves
        best_routes = cross_route_or_opt(best_routes, 15)
        # Final 2-opt with tighter tolerance
        if time.time() < deadline - 3.0:
            for i in range(len(best_routes)):
                best_routes[i] = two_opt(best_routes[i], 50, tol=1e-9)

    # Final validation
    all_served = []
    for route in best_routes:
        for c in route:
            if is_customer[c]:
                all_served.append(c)

    if len(all_served) != len(customers) or len(set(all_served)) != len(customers):
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
