"""EV routing with time windows and battery constraints."""
import numpy as np
import time
import math
import random
import sys

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_nodes = len(coords)

    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        total_dist = 0.0
        current_time = 0.0
        battery = battery_cap
        load = 0.0

        # Start from depot (node 0)
        prev = 0
        for node in route:
            d = dist[prev][node]
            total_dist += d
            battery -= d
            current_time += d / speed

            if battery < -1e-9:
                return False, total_dist

            if node in stations:
                # Charging station: wait to recharge
                charge_time = (battery_cap - battery) * recharge_rate
                current_time += charge_time
                battery = battery_cap
            else:
                # Customer
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist

                if current_time < ready[node]:
                    current_time = ready[node]
                if current_time > due[node] + 1e-9:
                    return False, total_dist
                current_time += service[node]

            prev = node

        # Return to depot
        d = dist[prev][0]
        total_dist += d
        battery -= d
        current_time += d / speed

        if battery < -1e-9:
            return False, total_dist
        if current_time > horizon + 1e-9:
            return False, total_dist

        return True, total_dist

    def route_distance(route):
        """Compute total distance of a route (depot to first, between, last to depot)."""
        total = 0.0
        prev = 0
        for node in route:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
        return total

    def simulate_route(route, need_details=False):
        """Simulate a route, return (feasible, total_dist, final_time, final_battery)."""
        total_dist = 0.0
        current_time = 0.0
        battery = battery_cap
        load = 0.0
        prev = 0

        for node in route:
            d = dist[prev][node]
            total_dist += d
            battery -= d
            current_time += d / speed

            if battery < -1e-9:
                return False, total_dist, current_time, battery

            if node in stations:
                charge_time = (battery_cap - battery) * recharge_rate
                current_time += charge_time
                battery = battery_cap
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist, current_time, battery
                if current_time < ready[node]:
                    current_time = ready[node]
                if current_time > due[node] + 1e-9:
                    return False, total_dist, current_time, battery
                current_time += service[node]

            prev = node

        d = dist[prev][0]
        total_dist += d
        battery -= d
        current_time += d / speed

        if battery < -1e-9 or current_time > horizon + 1e-9:
            return False, total_dist, current_time, battery

        return True, total_dist, current_time, battery

    # Nearest-neighbor construction
    def construct_nn(unvisited, start_node=0):
        """Build a route using nearest neighbor heuristic from start_node."""
        route = []
        current = start_node
        battery = battery_cap
        current_time = 0.0
        load = 0.0
        remaining = set(unvisited)

        while remaining:
            # Find nearest feasible customer
            best_node = None
            best_dist = float('inf')
            for node in remaining:
                d = dist[current][node]
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                new_time = current_time + d / speed
                new_load = load + demands[node]
                if new_load > capacity + 1e-9:
                    continue
                if new_time < ready[node]:
                    new_time = ready[node]
                if new_time > due[node] + 1e-9:
                    continue
                # Check if we can return to depot after this
                ret_d = dist[node][0]
                if new_battery - ret_d < -1e-9:
                    continue
                if new_time + service[node] + ret_d / speed > horizon + 1e-9:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_node = node

            if best_node is None:
                break

            route.append(best_node)
            battery = battery - dist[current][best_node]
            current_time += dist[current][best_node] / speed
            if current_time < ready[best_node]:
                current_time = ready[best_node]
            current_time += service[best_node]
            load += demands[best_node]
            remaining.remove(best_node)
            current = best_node

        return route

    def two_opt(route):
        """Apply 2-opt improvement to a route."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            if n < 2:
                break
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Try reversing segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _, _, _ = simulate_route(new_route)
                    if feasible:
                        old_cost = route_distance(route)
                        new_cost = route_distance(new_route)
                        if new_cost < old_cost - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
        return route

    def relocate(route, unvisited):
        """Try to relocate customers from route to unvisited or between routes."""
        # This is simplified - just try inserting unvisited into route
        pass

    def improve_route(route):
        """Improve a single route with 2-opt."""
        if len(route) < 2:
            return route
        return two_opt(route)

    # Main planning function
    def solve_instance():
        # Initial construction: nearest neighbor
        unvisited = set(customers)
        routes = []

        while unvisited and time.time() < deadline - 5.0:
            route = construct_nn(list(unvisited))
            if not route:
                # Fallback: put one customer per route
                for c in list(unvisited):
                    r = [c]
                    if simulate_route(r)[0]:
                        routes.append(r)
                        unvisited.remove(c)
                    else:
                        # Shouldn't happen but handle gracefully
                        routes.append(r)
                        unvisited.remove(c)
                break
            routes.append(route)
            for c in route:
                unvisited.discard(c)

        # Local search improvement
        if time.time() < deadline - 5.0:
            # 2-opt on each route
            for i in range(len(routes)):
                routes[i] = improve_route(routes[i])

        # Try to consolidate: remove empty routes, try merging
        if time.time() < deadline - 5.0:
            # Try to move customers between routes to reduce count
            for _ in range(3):  # a few passes
                changed = False
                for i in range(len(routes)):
                    if not routes[i]:
                        continue
                    for j in range(i + 1, len(routes)):
                        if not routes[j]:
                            continue
                        # Try moving each customer from route i to route j
                        for k in range(len(routes[i])):
                            c = routes[i][k]
                            new_route_i = routes[i][:k] + routes[i][k+1:]
                            # Try inserting c into route j at each position
                            for pos in range(len(routes[j]) + 1):
                                new_route_j = routes[j][:pos] + [c] + routes[j][pos:]
                                if simulate_route(new_route_j)[0]:
                                    # Check if remaining route i is still feasible
                                    if new_route_i and simulate_route(new_route_i)[0]:
                                        old_cost = route_distance(routes[i]) + route_distance(routes[j])
                                        new_cost = route_distance(new_route_i) + route_distance(new_route_j)
                                        if new_cost < old_cost - 1e-9:
                                            routes[i] = new_route_i
                                            routes[j] = new_route_j
                                            changed = True
                                            break
                            if changed:
                                break
                    if changed:
                        break
                if not changed:
                    break

        # Final 2-opt pass
        if time.time() < deadline - 5.0:
            for i in range(len(routes)):
                routes[i] = improve_route(routes[i])

        # Verify all customers served exactly once
        served = []
        for r in routes:
            served.extend(r)
        if set(served) != set(customers) or len(served) != len(customers):
            # Fallback: one customer per route
            return [[c] for c in customers]

        return routes

    # Run solver
    result = solve_instance()

    # Verify feasibility of final result
    for r in result:
        feasible, _, _, _ = simulate_route(r)
        if not feasible:
            # Fallback to trivial solution
            return [[c] for c in customers]

    return result
