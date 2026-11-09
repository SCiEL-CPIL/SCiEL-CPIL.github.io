"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

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
        remaining = set(customer_order)
        routes = []

        while remaining:
            # Start new route from depot
            route = []
            prev = 0
            t = 0.0
            bat = battery
            load = 0

            while remaining:
                # Find best next customer
                best_customer = None
                best_score = float('inf')

                for v in remaining:
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

                    # Score: distance + time penalty
                    score = d + (t_after - ready[v]) * 0.1

                    if score < best_score:
                        best_score = score
                        best_customer = v

                if best_customer is None:
                    break

                # Add customer
                v = best_customer
                d = dist[prev, v]
                bat -= d
                t += d / speed
                load += demands[v]

                if t < ready[v]:
                    t = ready[v]
                t += service[v]

                route.append(v)
                remaining.discard(v)
                prev = v

            if route:
                # Check if we can return to depot
                d = dist[prev, 0]
                if bat - d >= 0 and t + d / speed <= horizon:
                    routes.append(route)
                else:
                    # Can't complete this route, add to routes anyway
                    routes.append(route)

        return routes

    def two_opt(route):
        """Apply 2-opt to improve route order while maintaining feasibility."""
        if len(route) < 2:
            return route

        best = route[:]
        best_dist = dist[0, best[0]] + dist[best[-1], 0]
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                best_dist += dist[best[i], best[i + 1]]
                if j < len(best) - 1:
                    best_dist += dist[best[j], best[j + 1]]

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best) - 1):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(best)):
                    if time.time() > deadline:
                        break
                    # Try reversing segment i..j
                    new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                    feasible, d = check_route(new_route)
                    if feasible:
                        new_dist = total_route_distance([new_route])
                        if new_dist < best_dist - 1e-6:
                            best = new_route
                            best_dist = new_dist
                            improved = True
        return best

    def or_opt(routes):
        """Apply Or-opt moves: relocate one customer to a better position."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_dist = total_route_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for pos in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    v = routes[i][pos]
                    # Try removing v from route i
                    route_i_without = [x for x in routes[i] if x != v]

                    for k in range(len(routes)):
                        if time.time() > deadline:
                            break
                        for new_pos in range(len(routes[k]) + 1):
                            if time.time() > deadline:
                                break
                            # Create new routes
                            new_routes = []
                            for r in range(len(routes)):
                                if r == i:
                                    new_routes.append(route_i_without[:])
                                elif r == k:
                                    new_route = routes[k][:]
                                    new_route.insert(new_pos, v)
                                    new_routes.append(new_route)
                                else:
                                    new_routes.append(routes[r][:])

                            # Check feasibility
                            feasible = True
                            for route in new_routes:
                                f, _ = check_route(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def swap_opt(routes):
        """Try swapping two customers between routes."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_dist = total_route_distance(routes)

            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(len(routes[i])):
                    if time.time() > deadline:
                        break
                    for k in range(len(routes)):
                        if time.time() > deadline:
                            break
                        if k == i:
                            continue
                        for l in range(len(routes[k])):
                            if time.time() > deadline:
                                break
                            v1 = routes[i][j]
                            v2 = routes[k][l]

                            # Create new routes with swap
                            new_routes = []
                            for r in range(len(routes)):
                                if r == i:
                                    new_route = routes[r][:]
                                    new_route[j] = v2
                                    new_routes.append(new_route)
                                elif r == k:
                                    new_route = routes[r][:]
                                    new_route[l] = v1
                                    new_routes.append(new_route)
                                else:
                                    new_routes.append(routes[r][:])

                            # Check feasibility
                            feasible = True
                            for route in new_routes:
                                f, _ = check_route(route)
                                if not f:
                                    feasible = False
                                    break

                            if feasible:
                                new_dist = total_route_distance(new_routes)
                                if new_dist < current_dist - 1e-6:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # Initial solution: one customer per route (always feasible)
    best_routes = [[v] for v in customers]
    best_distance = sum(dist[0, v] * 2 for v in customers)

    # Try greedy construction with different orderings
    orderings = [
        sorted(customers, key=lambda v: due[v]),
        sorted(customers, key=lambda v: ready[v]),
        sorted(customers, key=lambda v: -demands[v]),
        sorted(customers, key=lambda v: dist[0, v]),
        customers  # natural order
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        greedy_routes = greedy_construct(order)
        # Validate greedy routes
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

    # Apply 2-opt to each route
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt(best_routes[i])

    # Apply Or-opt
    if time.time() < deadline:
        best_routes = or_opt(best_routes)

    # Apply swap-opt
    if time.time() < deadline:
        best_routes = swap_opt(best_routes)

    # Final 2-opt pass
    if time.time() < deadline:
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            best_routes[i] = two_opt(best_routes[i])

    return best_routes
# EVOLVE-BLOCK-END
