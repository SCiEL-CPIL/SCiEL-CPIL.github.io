# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt
from itertools import permutations


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = sqrt(dx * dx + dy * dy)
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate_route(route, D, demands, ready, due, service, stations, capacity, battery, speed, recharge_rate, horizon):
    """
    Simulate a route. Returns (is_valid, total_distance, total_time).
    """
    if not route:
        return True, 0.0, 0.0

    total_load = sum(demands[v] for v in route)
    if total_load > capacity + 1e-9:
        return False, 0.0, 0.0

    cur_node = 0
    cur_time = 0.0
    cur_bat = battery
    total_dist = 0.0

    for node in route:
        d = D[cur_node, node]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist, cur_time

        if node in stations:
            # Recharge
            energy = max(0.0, battery - cur_bat)
            cur_time += energy * recharge_rate
            cur_bat = battery
        else:
            # Customer
            if cur_time < ready[node]:
                cur_time = ready[node]
            if cur_time > due[node] + 1e-9:
                return False, total_dist, cur_time
            cur_time += service[node]

    # Return to depot
    d = D[cur_node, 0]
    cur_time += d / speed
    cur_bat -= d
    total_dist += d

    if cur_bat < -1e-9:
        return False, total_dist, cur_time
    if cur_time > horizon + 1e-9:
        return False, total_dist, cur_time

    return True, total_dist, cur_time


def _construct_routes(instance, D, deadline):
    """
    Construct initial routes using a greedy insertion heuristic.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    # Sort by due time for initial ordering
    customers.sort(key=lambda v: due[v])

    routes = []
    unassigned = set(customers)

    while unassigned:
        # Start a new route with the customer having the earliest due time
        start_node = min(unassigned, key=lambda v: due[v])
        route = [start_node]
        unassigned.remove(start_node)

        # Try to insert other customers
        changed = True
        while changed and time.time() < deadline:
            changed = False
            best_gain = -1
            best_node = -1
            best_pos = -1

            # For each unassigned customer, try inserting at each position
            for c in unassigned:
                # Check capacity
                current_load = sum(demands[v] for v in route)
                if current_load + demands[c] > capacity + 1e-9:
                    continue

                # Try inserting at each position (including before stations? No, stations are separate)
                # We only insert customers. Stations are handled during simulation/optimization if needed.
                # Actually, the route list contains customers. Stations are not in the route list in the output?
                # Wait, the task says: "Return a list of routes, each a list of node ids".
                # "stations may repeat".
                # The simulation function I wrote assumes `route` contains both customers and stations?
                # Let's re-read: "each a list of node ids; the depot is implicit at both ends."
                # "at a customer... at a station..."
                # So the route list CAN contain station ids.
                # However, my `_simulate_route` checks `if node in stations`.
                # So the route list should contain customers AND stations in the correct order.
                # My construction above only puts customers in `route`.
                # I need to handle stations.
                # For now, let's assume we can insert customers. Stations can be added later if battery is low.
                # Or, we can treat the route as a sequence of customers, and the simulator inserts stations automatically?
                # No, the simulator uses the explicit route list.
                # So if I want to visit a station, it must be in the list.
                # This makes the insertion heuristic more complex.
                # Simplification: Only insert customers. If battery runs out, the route is invalid.
                # We can try to add stations later if needed, or just rely on the fact that battery=300 is large enough
                # for most routes if we don't go too far.
                # Let's stick to inserting customers first.

                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    valid, dist, _ = _simulate_route(
                        new_route, D, demands, ready, due, service, stations,
                        capacity, battery, speed, recharge_rate, horizon
                    )
                    if valid:
                        # Calculate gain (reduction in distance compared to separate routes)
                        # Approximate gain: distance from current route's end to c + c to start of new route?
                        # Better: just minimize the new total distance.
                        # Since we are adding to an existing route, we want to minimize the added distance.
                        # Current route distance is not easily available without re-simulating the whole thing,
                        # but we can approximate the incremental cost.
                        # Or simply pick the insertion that results in the smallest total distance for the new route.
                        # Let's store the dist and pick the min.
                        if dist < best_gain or best_gain == -1:
                            best_gain = dist
                            best_node = c
                            best_pos = pos

            if best_node != -1:
                route = route[:best_pos] + [best_node] + route[best_pos:]
                unassigned.remove(best_node)
                changed = True

        routes.append(route)

    return routes


def _optimize_routes(routes, instance, D, deadline):
    """
    Improve routes using 2-opt and Or-opt within time limit.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    for r_idx in range(len(routes)):
        route = routes[r_idx]
        if len(route) < 2:
            continue

        improved = True
        while improved and time.time() < deadline:
            improved = False

            # 2-opt
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    valid, dist, _ = _simulate_route(
                        new_route, D, demands, ready, due, service, stations,
                        capacity, battery, speed, recharge_rate, horizon
                    )
                    if valid:
                        _, old_dist, _ = _simulate_route(
                            route, D, demands, ready, due, service, stations,
                            capacity, battery, speed, recharge_rate, horizon
                        )
                        if dist < old_dist - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break

            # Or-opt (move 1 or 2 nodes to a different position)
            if not improved:
                for i in range(len(route)):
                    for length in [1, 2]:
                        if i + length > len(route):
                            continue
                        segment = route[i:i + length]
                        rest = route[:i] + route[i + length:]
                        for j in range(len(rest) + 1):
                            new_route = rest[:j] + segment + rest[j:]
                            if new_route == route:
                                continue
                            valid, dist, _ = _simulate_route(
                                new_route, D, demands, ready, due, service, stations,
                                capacity, battery, speed, recharge_rate, horizon
                            )
                            if valid:
                                _, old_dist, _ = _simulate_route(
                                    route, D, demands, ready, due, service, stations,
                                    capacity, battery, speed, recharge_rate, horizon
                                )
                                if dist < old_dist - 1e-9:
                                    route = new_route
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        routes[r_idx] = route

    return routes


def plan(instance):
    start_time = time.time()
    # Leave 10 seconds for safety
    deadline = start_time + 110

    coords = np.array(instance["coords"])
    D = _dist_matrix(coords)

    # Phase 1: Construction
    routes = _construct_routes(instance, D, deadline)

    # Phase 2: Optimization
    if time.time() < deadline:
        routes = _optimize_routes(routes, instance, D, deadline)

    return routes
# EVOLVE-BLOCK-END
