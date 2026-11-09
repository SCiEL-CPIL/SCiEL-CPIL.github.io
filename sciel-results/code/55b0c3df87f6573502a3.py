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
        total_demand = 0
        for v in route:
            total_demand += demands[v]
        if total_demand > capacity:
            return False, 0.0

        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0
        
        # Local variables for speed
        dist_l = dist
        ready_l = ready
        due_l = due
        service_l = service
        speed_l = speed

        for v in route:
            d = dist_l[prev, v]
            bat -= d
            t += d / speed_l
            total_dist += d

            if bat < 0:
                return False, 0.0

            # At customer
            if t < ready_l[v]:
                t = ready_l[v]
            if t > due_l[v]:
                return False, 0.0
            t += service_l[v]
            prev = v

        # Return to depot
        d = dist_l[prev, 0]
        bat -= d
        t += d / speed_l
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

    # Fast 2-opt with O(1) delta computation using prefix distances
    def two_opt_route(route):
        """Apply 2-opt to a single route with O(1) delta computation."""
        if len(route) < 2:
            return route
        
        def build_prefix(route):
            """Build prefix distance array: prefix[i] = dist from depot to route[i]."""
            n = len(route)
            prefix = np.zeros(n + 1)
            prefix[0] = dist[0, route[0]]
            for i in range(1, n):
                prefix[i] = prefix[i - 1] + dist[route[i - 1], route[i]]
            return prefix
        
        def route_total(route, prefix):
            return prefix[-1] + dist[route[-1], 0]
        
        def delta_2opt(route, a, b, prefix):
            """Compute delta for reversing route[a:b+1]."""
            n = len(route)
            # Old edges: (a-1, a) and (b, b+1)
            old1 = dist[route[a - 1], route[a]] if a > 0 else dist[0, route[a]]
            old2 = dist[route[b], route[b + 1]] if b < n - 1 else dist[route[b], 0]
            # New edges: (a-1, b) and (a, b+1)
            new1 = dist[route[a - 1], route[b]] if a > 0 else dist[0, route[b]]
            new2 = dist[route[a], route[b + 1]] if b < n - 1 else dist[route[a], 0]
            return (new1 + new2) - (old1 + old2)
        
        def reverse_segment(route, a, b):
            """Reverse route[a:b+1] in-place and return new prefix."""
            route[a:b + 1] = route[a:b + 1][::-1]
            return build_prefix(route)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(route)
            prefix = build_prefix(route)
            for a in range(n - 1):
                if time.time() > deadline:
                    break
                for b in range(a + 1, n):
                    d = delta_2opt(route, a, b, prefix)
                    if d < -1e-6:
                        new_route = list(route)
                        new_route[a:b + 1] = new_route[a:b + 1][::-1]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            route = new_route
                            prefix = build_prefix(route)
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
        """Apply Or-opt moves to improve route using O(1) delta distance."""
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
                    
                    # Pre-calculate neighbors of the segment in original route
                    u = route[a-1] if a > 0 else 0
                    v = route[a+seg_len] if a+seg_len < n else 0
                    start_node = route[a]
                    end_node = route[a+seg_len-1]
                    
                    # Try inserting segment at each position in remaining
                    for pos in range(len(remaining) + 1):
                        if pos == a or pos == a + seg_len:
                            continue
                        
                        # Determine x, y (neighbors of insertion slot in original route)
                        if pos < a:
                            i = pos
                            x = route[i-1] if i > 0 else 0
                            y = route[i] if i < n else 0
                        else:
                            i = pos + seg_len
                            x = route[i-1] if i > 0 else 0
                            y = route[i] if i < n else 0
                            
                        # Calculate delta distance
                        old_cost = dist[u, start_node] + dist[end_node, v] + dist[x, y]
                        new_cost = dist[u, v] + dist[x, start_node] + dist[end_node, y]
                        
                        if new_cost < old_cost - 1e-6:
                            new_route = remaining[:pos] + segment + remaining[pos:]
                            feasible, _ = check_route(new_route)
                            if feasible:
                                route = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return route

    # Efficient relocate: remove node, try all insertions with O(1) delta
    def relocate_search(routes, max_passes=5):
        """Remove-and-reinsert with fast delta computation."""
        routes = [list(r) for r in routes]
        
        def route_dist(r):
            if not r:
                return 0.0
            d = dist[0, r[0]]
            for i in range(len(r) - 1):
                d += dist[r[i], r[i + 1]]
            d += dist[r[-1], 0]
            return d
        
        for _ in range(max_passes):
            if time.time() > deadline:
                break
            improved = False
            current_total = sum(route_dist(r) for r in routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    v = routes[i][j]
                    # Remove v from route i
                    route_i = routes[i]
                    # Delta of removing v from route i
                    if len(route_i) == 1:
                        old_i_dist = dist[0, v] * 2
                        new_i_dist = 0.0
                    else:
                        prev_node = route_i[j - 1] if j > 0 else 0
                        next_node = route_i[j + 1] if j < len(route_i) - 1 else 0
                        old_i_dist = dist[prev_node, v] + dist[v, next_node]
                        new_i_dist = dist[prev_node, next_node]
                    
                    # Check feasibility of route i without v
                    route_i_without = route_i[:j] + route_i[j + 1:]
                    if route_i_without:
                        f_i, _ = check_route(route_i_without)
                        if not f_i:
                            continue
                    
                    # Try inserting v into each other route
                    for k in range(len(routes)):
                        if k == i and len(route_i_without) == 0:
                            continue
                        if time.time() > deadline:
                            break
                        route_k = routes[k] if k != i else route_i_without
                        n_k = len(route_k)
                        
                        for pos in range(n_k + 1):
                            # Skip same position
                            if k == i and pos == j:
                                continue
                            if k == i and pos == j + 1:
                                continue
                            
                            # Delta for inserting v at pos in route_k
                            if n_k == 0:
                                new_k_dist = dist[0, v] * 2
                            else:
                                prev_k = route_k[pos - 1] if pos > 0 else 0
                                next_k = route_k[pos] if pos < n_k else 0
                                new_k_dist = dist[prev_k, v] + dist[v, next_k]
                                old_k_segment = dist[prev_k, next_k]
                                # But we need to account for the full route change
                                # The delta is: (dist[prev_k,v] + dist[v,next_k]) - dist[prev_k,next_k]
                                # plus if route_k was empty before, add the depot legs
                            
                            # Full delta
                            if k != i:
                                if n_k == 0:
                                    # Route k was empty, now has just v
                                    delta_k = dist[0, v] * 2
                                else:
                                    prev_k = route_k[pos - 1] if pos > 0 else 0
                                    next_k = route_k[pos] if pos < n_k else 0
                                    delta_k = dist[prev_k, v] + dist[v, next_k] - dist[prev_k, next_k]
                            else:
                                # Inserting back into same route (different position)
                                if len(route_i_without) == 0:
                                    delta_k = dist[0, v] * 2
                                else:
                                    rw = route_i_without
                                    prev_k = rw[pos - 1] if pos > 0 else 0
                                    next_k = rw[pos] if pos < len(rw) else 0
                                    delta_k = dist[prev_k, v] + dist[v, next_k] - dist[prev_k, next_k]
                            
                            total_delta = delta_k + (new_i_dist - old_i_dist)
                            
                            if total_delta < -1e-6:
                                # Build new routes
                                new_routes = [list(r) for r in routes]
                                if k != i:
                                    new_routes[i] = route_i_without
                                    new_routes[k] = route_k[:pos] + [v] + route_k[pos:]
                                else:
                                    rw = route_i_without
                                    new_routes[i] = rw[:pos] + [v] + rw[pos:]
                                
                                # Verify feasibility
                                f_ok = True
                                for idx in range(len(new_routes)):
                                    if new_routes[idx]:
                                        f, _ = check_route(new_routes[idx])
                                        if not f:
                                            f_ok = False
                                            break
                                if f_ok:
                                    routes = new_routes
                                    current_total += total_delta
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
        return [r for r in routes if r]

    # Apply relocate search
    if time.time() < deadline:
        best_routes = relocate_search(best_routes, max_passes=5)

    # Cross-route swap with O(1) delta computation
    if time.time() < deadline:
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_routes)):
                    if time.time() > deadline:
                        break
                    ri = best_routes[i]
                    rj = best_routes[j]
                    for a in range(len(ri)):
                        if time.time() > deadline:
                            break
                        for b in range(len(rj)):
                            vi = ri[a]
                            vj = rj[b]
                            if vi == vj:
                                continue
                            
                            # Compute delta for swapping vi and vj
                            # Route i: remove vi at pos a, insert vj at pos a
                            if len(ri) == 1:
                                delta_i = dist[0, vj] * 2 - dist[0, vi] * 2
                            else:
                                prev_i = ri[a - 1] if a > 0 else 0
                                next_i = ri[a + 1] if a < len(ri) - 1 else 0
                                delta_i = (dist[prev_i, vj] + dist[vj, next_i]) - (dist[prev_i, vi] + dist[vi, next_i])
                            
                            # Route j: remove vj at pos b, insert vi at pos b
                            if len(rj) == 1:
                                delta_j = dist[0, vi] * 2 - dist[0, vj] * 2
                            else:
                                prev_j = rj[b - 1] if b > 0 else 0
                                next_j = rj[b + 1] if b < len(rj) - 1 else 0
                                delta_j = (dist[prev_j, vi] + dist[vi, next_j]) - (dist[prev_j, vj] + dist[vj, next_j])
                            
                            total_delta = delta_i + delta_j
                            if total_delta < -1e-6:
                                new_ri = list(ri)
                                new_rj = list(rj)
                                new_ri[a] = vj
                                new_rj[b] = vi
                                
                                f_i, _ = check_route(new_ri)
                                if not f_i:
                                    continue
                                f_j, _ = check_route(new_rj)
                                if not f_j:
                                    continue
                                
                                best_routes[i] = new_ri
                                best_routes[j] = new_rj
                                improved_swap = True
                                break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish: multiple passes of 2-opt + or-opt + relocate
    for polish_pass in range(3):
        if time.time() > deadline:
            break
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

        if time.time() > deadline:
            break
        best_routes = relocate_search(best_routes, max_passes=3)

        if time.time() > deadline:
            break
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = or_opt_route(best_routes[i])

        if time.time() > deadline:
            break
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt_route(best_routes[i])

    # Remove empty routes
    best_routes = [r for r in best_routes if r]

    return best_routes
# EVOLVE-BLOCK-END
