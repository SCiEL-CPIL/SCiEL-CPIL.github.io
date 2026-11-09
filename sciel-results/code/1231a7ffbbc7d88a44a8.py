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

    def greedy_construct(customer_order):
        """Construct routes using greedy nearest neighbor with time windows."""
        remaining = list(customer_order)
        remaining_set = set(remaining)
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
                remaining_set.discard(v)
                prev = v

            if route:
                # Check if we can return to depot
                d = dist[prev, 0]
                if bat - d >= 0 and t + d / speed <= horizon:
                    routes.append(route)
                else:
                    # Can't complete this route; take what we have and start new
                    routes.append(route)

        return routes

    def greedy_construct_insertion(customer_order):
        """Construct routes by inserting customers at best position in existing routes."""
        remaining = list(customer_order)
        routes = []

        while remaining:
            # Try to insert into an existing route
            inserted = False
            best_total_cost = float('inf')
            best_route_idx = -1
            best_pos = -1
            best_customer = None

            for idx, v in enumerate(remaining):
                for r_idx, route in enumerate(routes):
                    for pos in range(len(route) + 1):
                        # Compute cost of inserting v at pos in route
                        # New route: route[:pos] + [v] + route[pos:]
                        # Distance change:
                        # Remove: dist[route[pos-1] if pos>0 else 0, route[pos] if pos<len else 0]
                        # Add: dist[route[pos-1] if pos>0 else 0, v] + dist[v, route[pos] if pos<len else 0]
                        prev_node = route[pos - 1] if pos > 0 else 0
                        next_node = route[pos] if pos < len(route) else 0

                        d_prev = dist[prev_node, v]
                        d_next = dist[v, next_node]
                        d_old = dist[prev_node, next_node]

                        # Check battery feasibility for the new segment
                        # We need to verify the whole route is still feasible
                        new_route = route[:pos] + [v] + route[pos:]

                        # Quick checks
                        total_demand = sum(demands[x] for x in new_route)
                        if total_demand > capacity:
                            continue

                        # Full feasibility check
                        feasible, _ = check_route(new_route)
                        if not feasible:
                            continue

                        cost = d_prev + d_next - d_old
                        if cost < best_total_cost:
                            best_total_cost = cost
                            best_route_idx = r_idx
                            best_pos = pos
                            best_customer = v

            if best_customer is not None:
                route = routes[best_route_idx]
                route.insert(best_pos, best_customer)
                remaining.remove(best_customer)
                inserted = True

            if not inserted:
                # Start a new route with the first remaining customer
                v = remaining[0]
                remaining.remove(v)
                routes.append([v])

        return routes

    # Initial solution: one customer per route (always feasible)
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

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
        all_customers = []
        valid = True
        total_dist = 0.0
        for route in greedy_routes:
            feasible, d = check_route(route)
            if not feasible:
                valid = False
                break
            all_customers.extend(route)
            total_dist += d

        if valid and set(all_customers) == set(customers) and len(all_customers) == len(set(all_customers)):
            if total_dist < best_distance:
                best_routes = greedy_routes
                best_distance = total_dist

    # Try insertion-based construction
    for ordering in orderings:
        if time.time() > deadline:
            break
        ins_routes = greedy_construct_insertion(ordering)
        all_customers = []
        valid = True
        total_dist = 0.0
        for route in ins_routes:
            feasible, d = check_route(route)
            if not feasible:
                valid = False
                break
            all_customers.extend(route)
            total_dist += d

        if valid and set(all_customers) == set(customers) and len(all_customers) == len(set(all_customers)):
            if total_dist < best_distance:
                best_routes = ins_routes
                best_distance = total_dist

    # Remove-and-reinsert local search (more effective than simple move)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_dist = total_route_distance(best_routes)

        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            for j in range(len(best_routes[i])):
                if time.time() > deadline:
                    break
                v = best_routes[i][j]
                # Remove v from route i
                route_i_without = [x for x in best_routes[i] if x != v]
                if not route_i_without:
                    # Route becomes empty
                    new_routes = [r for idx, r in enumerate(best_routes) if idx != i]
                else:
                    new_routes = list(best_routes)
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

                        # Compute new total distance
                        new_routes[k] = new_route_k
                        new_dist = total_route_distance(new_routes)
                        if new_dist < current_dist - 1e-6:
                            best_routes = new_routes
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

    # 2-opt within each route (multiple passes)
    for _ in range(8):
        if time.time() > deadline:
            break
        improved_2opt = False
        for i in range(len(best_routes)):
            route = best_routes[i]
            if len(route) < 2:
                continue
            for a in range(len(route)):
                if time.time() > deadline:
                    break
                for b in range(a + 1, len(route)):
                    # Compute boundary distances
                    old_boundary = 0.0
                    if a > 0:
                        old_boundary += dist[route[a - 1], route[a]]
                    else:
                        old_boundary += dist[0, route[a]]
                    if b < len(route) - 1:
                        old_boundary += dist[route[b], route[b + 1]]
                    else:
                        old_boundary += dist[route[b], 0]

                    new_boundary = 0.0
                    if a > 0:
                        new_boundary += dist[route[a - 1], route[b]]
                    else:
                        new_boundary += dist[0, route[b]]
                    if b < len(route) - 1:
                        new_boundary += dist[route[a], route[b + 1]]
                    else:
                        new_boundary += dist[route[a], 0]

                    if new_boundary < old_boundary - 1e-6:
                        new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                        feasible, _ = check_route(new_route)
                        if feasible:
                            best_routes[i] = new_route
                            improved_2opt = True
                            # Restart 2-opt for this route
                            route = best_routes[i]
                            break
                if improved_2opt:
                    break
            if improved_2opt:
                break
        if not improved_2opt:
            break

    # Cross-route swap local search
    if time.time() < deadline:
        improved_swap = True
        while improved_swap and time.time() < deadline:
            improved_swap = False
            current_dist = total_route_distance(best_routes)
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best_routes)):
                    if i == j or time.time() > deadline:
                        continue
                    for a in range(len(best_routes[i])):
                        if time.time() > deadline:
                            break
                        for b in range(len(best_routes[j])):
                            vi = best_routes[i][a]
                            vj = best_routes[j][b]
                            # Swap
                            new_routes = []
                            for r in range(len(best_routes)):
                                if r == i:
                                    nr = best_routes[i][:]
                                    nr[a] = vj
                                    new_routes.append(nr)
                                elif r == j:
                                    nr = best_routes[j][:]
                                    nr[b] = vi
                                    new_routes.append(nr)
                                else:
                                    new_routes.append(best_routes[r][:])

                            feasible = True
                            for route in new_routes:
                                f, _ = check_route(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    best_routes = new_routes
                                    current_dist = new_dist
                                    improved_swap = True
                                    break
                        if improved_swap:
                            break
                    if improved_swap:
                        break
                if improved_swap:
                    break

    # Final polish: another round of remove-reinsert + 2-opt
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_dist = total_route_distance(best_routes)
            for i in range(len(best_routes)):
                if time.time() > deadline:
                    break
                for j in range(len(best_routes[i])):
                    if time.time() > deadline:
                        break
                    v = best_routes[i][j]
                    route_i_without = [x for x in best_routes[i] if x != v]
                    if not route_i_without:
                        new_routes = [r for idx, r in enumerate(best_routes) if idx != i]
                    else:
                        new_routes = list(best_routes)
                        new_routes[i] = route_i_without

                    if route_i_without:
                        feasible_i, _ = check_route(route_i_without)
                        if not feasible_i:
                            continue
                    else:
                        feasible_i = True

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
                                best_routes = new_routes
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

        # Final 2-opt
        for _ in range(3):
            if time.time() > deadline:
                break
            improved_2opt = False
            for i in range(len(best_routes)):
                route = best_routes[i]
                if len(route) < 2:
                    continue
                for a in range(len(route)):
                    if time.time() > deadline:
                        break
                    for b in range(a + 1, len(route)):
                        old_boundary = 0.0
                        if a > 0:
                            old_boundary += dist[route[a - 1], route[a]]
                        else:
                            old_boundary += dist[0, route[a]]
                        if b < len(route) - 1:
                            old_boundary += dist[route[b], route[b + 1]]
                        else:
                            old_boundary += dist[route[b], 0]

                        new_boundary = 0.0
                        if a > 0:
                            new_boundary += dist[route[a - 1], route[b]]
                        else:
                            new_boundary += dist[0, route[b]]
                        if b < len(route) - 1:
                            new_boundary += dist[route[a], route[b + 1]]
                        else:
                            new_boundary += dist[route[a], 0]

                        if new_boundary < old_boundary - 1e-6:
                            new_route = route[:a] + route[a:b + 1][::-1] + route[b + 1:]
                            feasible, _ = check_route(new_route)
                            if feasible:
                                best_routes[i] = new_route
                                improved_2opt = True
                                route = best_routes[i]
                                break
                    if improved_2opt:
                        break
                if improved_2opt:
                    break
            if not improved_2opt:
                break

    return best_routes
# EVOLVE-BLOCK-END
