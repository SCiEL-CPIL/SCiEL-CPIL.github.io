"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix with NumPy
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            bat -= d
            t += d / speed
            total_dist += d

            if bat < 0:
                return False, 0.0

            # At customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v]:
                return False, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev, 0]
        bat -= d
        t += d / speed
        total_dist += d

        if bat < 0:
            return False, 0.0
        if t > horizon:
            return False, 0.0

        return True, total_dist

    def total_route_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            total += dist[0, route[0]]
            for i in range(len(route) - 1):
                total += dist[route[i], route[i + 1]]
            total += dist[route[-1], 0]
        return total

    def savings_construct():
        """Clarke-Wright savings based construction."""
        # Start with each customer in its own route
        routes = [[v] for v in customers]
        route_of = {}
        for i, v in enumerate(customers):
            route_of[v] = i

        # Compute savings
        savings = []
        for i in range(n_customers):
            for j in range(i + 1, n_customers):
                vi, vj = i + 1, j + 1
                s = dist[0, vi] + dist[0, vj] - dist[vi, vj]
                savings.append((s, vi, vj))

        savings.sort(reverse=True)

        for s, vi, vj in savings:
            if time.time() > deadline:
                break
            ri = route_of[vi]
            rj = route_of[vj]
            if ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check if vi is at the end of route_i or vj is at the end of route_j
            # We can merge if:
            # - vi is last in route_i and vj is first in route_j
            # - vi is first in route_i and vj is last in route_j
            # etc.

            can_merge = False
            new_route = None

            # Option 1: append route_j to route_i (vi must be last in i, vj first in j)
            if route_i[-1] == vi and route_j[0] == vj:
                candidate = route_i + route_j
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate
            # Option 2: append route_i to route_j
            elif route_j[-1] == vj and route_i[0] == vi:
                candidate = route_j + route_i
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate
            # Option 3: reverse route_i then append
            elif route_i[0] == vi and route_j[0] == vj:
                candidate = route_i[::-1] + route_j
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate
            # Option 4: reverse route_j then append
            elif route_i[-1] == vi and route_j[-1] == vj:
                candidate = route_i + route_j[::-1]
                if sum(demands[x] for x in candidate) <= capacity:
                    f, _ = check_route(candidate)
                    if f:
                        can_merge = True
                        new_route = candidate

            if can_merge:
                # Remove route_j, update route_i
                routes[ri] = new_route
                routes[rj] = []
                for v in new_route:
                    route_of[v] = ri
                # Clean up empty routes
                routes = [r for r in routes if r]
                # Rebuild route_of
                for i, r in enumerate(routes):
                    for v in r:
                        route_of[v] = i

        return [r for r in routes if r]

    def greedy_construct(customer_order):
        """Construct routes using greedy nearest neighbor with time windows."""
        remaining = list(customer_order)
        routes = []

        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0

            while remaining:
                best_customer = None
                best_score = float('inf')
                best_idx = -1

                for idx, v in enumerate(remaining):
                    d = dist[prev, v]
                    bat_after = bat - d
                    t_after = t + d / speed

                    if bat_after < 0:
                        continue
                    if load + demands[v] > capacity:
                        continue

                    # Check time window
                    if t_after < ready[v]:
                        t_after = ready[v]
                    if t_after > due[v]:
                        continue

                    # Score: distance + small time penalty
                    score = d + (t_after - ready[v]) * 0.05

                    if score < best_score:
                        best_score = score
                        best_customer = v
                        best_idx = idx

                if best_customer is None:
                    break

                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]

                if t < ready[v]:
                    t = ready[v]
                t += service[v]

                route.append(v)
                remaining.pop(best_idx)
                prev = v

            if route:
                routes.append(route)

        return routes

    def validate_routes(routes):
        """Check if routes are valid: all customers served exactly once, all feasible."""
        all_customers = []
        for route in routes:
            feasible, _ = check_route(route)
            if not feasible:
                return False
            all_customers.extend(route)
        if len(all_customers) != len(set(all_customers)):
            return False
        if set(all_customers) != set(customers):
            return False
        return True

    # Initial solution: one customer per route (always feasible)
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try savings construction
    if time.time() < deadline:
        sw_routes = savings_construct()
        if validate_routes(sw_routes):
            d = total_route_distance(sw_routes)
            if d < best_distance:
                best_routes = sw_routes
                best_distance = d

    # Try greedy construction with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        sorted(customers, key=lambda v: (due[v], dist[0, v])),
    ]

    for ordering in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(ordering)
        if validate_routes(greedy_routes):
            d = total_route_distance(greedy_routes)
            if d < best_distance:
                best_routes = greedy_routes
                best_distance = d

    # Vectorized 2-opt within each route
    def two_opt_route(route):
        """Apply 2-opt to a single route until no improvement."""
        if len(route) < 2:
            return route
        n = len(route)
        # Precompute full route distances for O(1) delta computation
        def route_dist(r):
            if not r:
                return 0.0
            d = dist[0, r[0]]
            for i in range(len(r) - 1):
                d += dist[r[i], r[i + 1]]
            d += dist[r[-1], 0]
            return d
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    # Compute boundary distances
                    old_boundary = 0.0
                    if a > 0:
                        old_boundary += dist[route[a - 1], route[a]]
                    else:
                        old_boundary += dist[0, route[a]]
                    if b < n - 1:
                        old_boundary += dist[route[b], route[b + 1]]
                    else:
                        old_boundary += dist[route[b], 0]

                    new_boundary = 0.0
                    if a > 0:
                        new_boundary += dist[route[a - 1], route[b]]
                    else:
                        new_boundary += dist[0, route[b]]
                    if b < n - 1:
                        new_boundary += dist[route[a], route[b + 1]]
                    else:
                        new_boundary += dist[route[a], 0]

                    if new_boundary < old_boundary - 1e-6:
                        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    # Apply 2-opt to all routes
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])
    
    # Or-opt: move a single node or segment within a route
    def or_opt_route(route):
        """Apply Or-opt moves to improve route."""
        if len(route) < 2:
            return route
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(route)
            for a in range(n):
                if time.time() > deadline:
                    break
                for seg_len in [1, 2, 3]:
                    if a + seg_len > n:
                        continue
                    segment = route[a:a + seg_len]
                    # Remove segment
                    remaining = route[:a] + route[a + seg_len:]
                    if not remaining:
                        continue
                    # Try inserting segment at each position in remaining
                    for pos in range(len(remaining) + 1):
                        if pos == a or pos == a + seg_len:
                            continue
                        new_route = remaining[:pos] + segment + remaining[pos:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            old_d = dist[0, route[0]] + sum(dist[route[i], route[i+1]] for i in range(len(route)-1)) + dist[route[-1], 0]
                            new_d = dist[0, new_route[0]] + sum(dist[new_route[i], new_route[i+1]] for i in range(len(new_route)-1)) + dist[new_route[-1], 0]
                            if new_d < old_d - 1e-6:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    # Remove-and-reinsert local search with fast checks
    def relocate_search(routes, max_passes=10):
        """Remove-and-reinsert local search."""
        routes = [list(r) for r in routes]
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_dist = total_route_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    v = routes[i][j]
                    # Remove v from route i
                    route_i_without = routes[i][:j] + routes[i][j+1:]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(routes) if idx != i]
                    else:
                        new_routes = list(routes)
                        new_routes[i] = route_i_without

                    # Check feasibility of modified route i
                    if route_i_without:
                        feasible_i, _ = check_route(route_i_without)
                        if not feasible_i:
                            continue
                    else:
                        feasible_i = True

                    # Try inserting v into each position of each other route
                    for k in range(len(new_routes)):
                        if time.time() > deadline:
                            break
                        route_k = new_routes[k]
                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [v] + route_k[pos:]
                            feasible, _ = check_route(new_route_k)
                            if not feasible:
                                continue

                            new_routes[k] = new_route_k
                            new_dist = total_route_distance(new_routes)
                            if new_dist < current_dist - 1e-6:
                                routes = new_routes
                                current_dist = new_dist
                                improved = True
                                break
                            else:
                                new_routes[k] = route_k
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return routes

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # Cross-route swap local search (more efficient: check feasibility of only affected routes)
    if time.time() < deadline:
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            current_dist = total_route_distance(best_routes)
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_routes)):
                    if time.time() > deadline:
                        continue
                    for a in range(len(best_routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(best_routes[j])):
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            if vi == vj:
                                continue
                            # Create modified routes
                            new_route_i = best_routes[i][:]
                            new_route_j = best_routes[j][:]
                            new_route_i[a] = vj
                            new_route_j[b] = vi

                            # Check feasibility of only the two modified routes
                            f_i, _ = check_route(new_route_i)
                            if not f_i:
                                continue
                            f_j, _ = check_route(new_route_j)
                            if not f_j:
                                continue

                            # Compute new total distance
                            new_dist = current_dist
                            # Remove old contributions from routes i and j
                            for r in [i, j]:
                                route = best_routes[r]
                                if route:
                                    old_contrib = dist[0, route[0]]
                                    for k in range(len(route) - 1):
                                        old_contrib += dist[route[k], route[k + 1]]
                                    old_contrib += dist[route[-1], 0]
                                    new_dist -= old_contrib
                            # Add new contributions
                            for r in [i, j]:
                                route = [new_route_i if r == i else new_route_j][0]
                                if route:
                                    new_contrib = dist[0, route[0]]
                                    for k in range(len(route) - 1):
                                        new_contrib += dist[route[k], route[k + 1]]
                                    new_contrib += dist[route[-1], 0]
                                    new_dist += new_contrib

                            if new_dist < current_dist - 1e-6:
                                best_routes[i] = new_route_i
                                best_routes[j] = new_route_j
                                current_dist = new_dist
                                improved_swap = True
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish: 2-opt + or-opt + relocate
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = or_opt_route(best_routes[i])

        if time.time() < deadline:
            best_routes = relocate_search(best_routes, max_passes=3)

        # Final 2-opt
        if time.time() < deadline:
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
