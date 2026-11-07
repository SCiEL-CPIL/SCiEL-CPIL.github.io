"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import combinations


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
            dist[i][j] = d
            dist[j][i] = d

    customers = list(range(1, n_customers + 1))

    def route_distance(route):
        """Compute total Euclidean distance of a route including depot legs."""
        if not route:
            return 0.0
        total = dist[0][route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i]][route[i+1]]
        total += dist[route[-1]][0]
        return total

    def simulate_route(route):
        """
        Simulate a route and return (feasible, total_dist, end_time, end_battery).
        Battery starts full at depot, time starts at 0.
        """
        if not route:
            return True, 0.0, 0.0, battery_cap

        t = 0.0
        batt = battery_cap
        load = 0

        # Go from depot to first node
        d = dist[0][route[0]]
        t += d / speed
        batt -= d

        if batt < -1e-9:
            return False, 0.0, 0.0, 0.0

        for node in route:
            if node in stations:
                # Charging station: recharge
                energy_needed = battery_cap - batt
                if energy_needed > 0:
                    t += energy_needed * recharge_rate
                batt = battery_cap
            else:
                # Customer
                # Check time window
                if t > due[node] + 1e-9:
                    return False, 0.0, 0.0, 0.0
                t = max(t, ready[node])
                # Service
                t += service[node]
                # Demand
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0, 0.0, 0.0

            # Check battery is non-negative
            if batt < -1e-9:
                return False, 0.0, 0.0, 0.0

        # Return to depot
        d = dist[route[-1]][0]
        t += d / speed
        batt -= d

        if batt < -1e-9:
            return False, 0.0, 0.0, 0.0
        if t > horizon + 1e-9:
            return False, 0.0, 0.0, 0.0

        total_dist = dist[0][route[0]]
        for i in range(len(route) - 1):
            total_dist += dist[route[i]][route[i+1]]
        total_dist += dist[route[-1]][0]

        return True, total_dist, t, batt

    def is_feasible(route):
        feasible, _, _, _ = simulate_route(route)
        return feasible

    def greedy_construct(remaining_customers):
        """
        Greedy nearest-neighbor construction with feasibility checks.
        Returns a list of routes.
        """
        routes = []
        unvisited = set(remaining_customers)

        while unvisited:
            # Start a new route
            route = []
            current_node = 0
            t = 0.0
            batt = battery_cap
            load = 0

            while unvisited:
                # Find best next customer to visit
                best_node = None
                best_dist = float('inf')
                best_new_t = None
                best_new_batt = None

                for c in unvisited:
                    d = dist[current_node][c]
                    new_batt = batt - d
                    if new_batt < -1e-9:
                        continue
                    new_t = t + d / speed
                    if c in stations:
                        # Shouldn't happen since we only visit customers here
                        continue
                    # Time window check
                    if new_t > due[c] + 1e-9:
                        continue
                    new_t = max(new_t, ready[c]) + service[c]
                    new_load = load + demands[c]
                    if new_load > capacity + 1e-9:
                        continue
                    # Check if we can return to depot from c
                    d_back = dist[c][0]
                    if new_batt - d_back < -1e-9:
                        continue
                    if new_t + d_back / speed > horizon + 1e-9:
                        continue

                    if d < best_dist:
                        best_dist = d
                        best_node = c
                        best_new_t = new_t
                        best_new_batt = new_batt

                if best_node is None:
                    break

                # Move to best_node
                d = dist[current_node][best_node]
                t += d / speed
                batt -= d
                t = max(t, ready[best_node])
                t += service[best_node]
                load += demands[best_node]
                route.append(best_node)
                unvisited.discard(best_node)
                current_node = best_node

            if route:
                # Verify feasibility
                feasible, total_dist, end_t, end_batt = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Fallback: one customer per route
                    for c in route:
                        if c in unvisited:
                            unvisited.add(c)
                    # Actually route customers were removed from unvisited, so add back
                    # Let's just handle this by putting them back
                    for c in route:
                        unvisited.add(c)
                    # Now serve them one by one
                    for c in sorted(route, key=lambda x: due[x]):
                        single = [c]
                        feasible, _, _, _ = simulate_route(single)
                        if feasible:
                            routes.append(single)
                            unvisited.discard(c)

        return routes

    def two_opt_improve(routes):
        """Apply 2-opt local search to improve routes while maintaining feasibility."""
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                if len(route) < 2:
                    continue
                # Try all 2-opt moves
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if new_route == route:
                            continue
                        old_dist = route_distance(route)
                        new_dist = route_distance(new_route)
                        if new_dist < old_dist - 1e-9:
                            feasible, _, _, _ = simulate_route(new_route)
                            if feasible:
                                routes[r_idx] = new_route
                                improved = True

        return routes

    def relocate_improve(routes):
        """Try to relocate customers between routes to reduce total distance."""
        improved = True
        max_iterations = 20
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for r1 in range(len(routes)):
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for i, c in enumerate(routes[r1]):
                        # Try inserting c into route r2 at each position
                        for j in range(len(routes[r2]) + 1):
                            new_route1 = routes[r1][:i] + routes[r1][i+1:]
                            new_route2 = routes[r2][:j] + [c] + routes[r2][j:]
                            if not new_route1 and not new_route2:
                                continue
                            feasible1, dist1, _, _ = simulate_route(new_route1) if new_route1 else (True, 0.0, 0.0, 0.0)
                            feasible2, dist2, _, _ = simulate_route(new_route2)
                            if feasible1 and feasible2:
                                old_total = route_distance(routes[r1]) + route_distance(routes[r2])
                                new_total = dist1 + dist2
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
            # Remove empty routes
            routes = [r for r in routes if r]

        return routes

    # Initial construction
    best_routes = greedy_construct(customers)

    # Local search improvements
    best_routes = two_opt_improve(best_routes)
    best_routes = relocate_improve(best_routes)

    # Check time and potentially do more iterations
    if time.time() < deadline - 30:
        best_routes = two_opt_improve(best_routes)
        best_routes = relocate_improve(best_routes)

    if time.time() < deadline - 15:
        best_routes = two_opt_improve(best_routes)

    # Final verification
    final_routes = []
    for route in best_routes:
        feasible, _, _, _ = simulate_route(route)
        if feasible:
            final_routes.append(route)
        else:
            # Fallback: serve each customer individually
            for c in route:
                single = [c]
                feasible, _, _, _ = simulate_route(single)
                if feasible:
                    final_routes.append(single)

    # Ensure all customers are served
    served = set()
    for route in final_routes:
        for c in route:
            if c not in stations:
                served.add(c)

    for c in customers:
        if c not in served:
            final_routes.append([c])

    return final_routes
# EVOLVE-BLOCK-END
