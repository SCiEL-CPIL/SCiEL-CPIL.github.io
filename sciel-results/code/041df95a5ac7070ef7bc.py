"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import heapq
import random


def _build_dist_matrix(coords):
    """Build Euclidean distance matrix using NumPy."""
    n = len(coords)
    c = np.array(coords, dtype=np.float64)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """
    Simulate a single route. Returns (total_distance, is_valid).
    route: list of node ids (customers and/or stations), depot implicit at start/end.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_dist = 0.0
    t = 0.0
    batt = battery_max
    load = 0
    prev = 0  # depot

    for node in route:
        d = dist[prev][node]
        total_dist += d
        batt -= d
        t += d / speed

        if batt < -1e-9:
            return total_dist, False

        load += demands[node]
        if load > capacity:
            return total_dist, False

        if node in stations_set:
            # Recharge
            if batt < 0:
                batt = 0.0
            energy = battery_max - batt
            t += energy * recharge_rate
            batt = battery_max
        else:
            # Customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False
            t += service[node]

        prev = node

    # Return to depot
    d = dist[prev][0]
    total_dist += d
    batt -= d
    t += d / speed

    if batt < -1e-9:
        return total_dist, False
    if t > horizon + 1e-9:
        return total_dist, False

    return total_dist, True


def _validate_solution(routes, dist, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity):
    """Validate that all customers are served exactly once and all routes are feasible."""
    served = set()
    total_dist = 0.0
    for r in routes:
        d, valid = _simulate_route(r, dist, instance, n_customers, stations_set,
                                   battery_max, speed, recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_dist += d
        for node in r:
            if node in stations_set:
                continue
            if node in served:
                return False, 0.0
            served.add(node)

    all_customers = set(range(1, n_customers + 1))
    if served != all_customers:
        return False, 0.0

    return True, total_dist


def _compute_total_distance(routes, dist):
    """Compute total distance of all routes."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        prev = 0
        for node in r:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
    return total


def _route_feasible(route, dist, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """Quick check if a route is feasible."""
    _, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                               battery_max, speed, recharge_rate, horizon, capacity)
    return valid


def _dijkstra_route_build(remaining_customers, dist, instance, n_customers,
                          stations_set, battery_max, speed, recharge_rate,
                          horizon, capacity, max_route_length=15):
    """
    Build a single route using Dijkstra on state space (node, battery, visited_mask).
    Returns (route, total_distance) or (None, inf) if no feasible route found.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    if not remaining_customers:
        return [], 0.0

    # State: (node, battery, visited_frozenset)
    # We use a priority queue: (distance, node, battery, visited)
    # Start at depot with full battery, no visited customers
    start_state = (0, battery_max, frozenset())
    # Priority queue: (distance, node, battery, visited)
    pq = [(0.0, 0, battery_max, frozenset())]
    # Best distance seen for each state
    best = {}
    best[start_state] = 0.0
    # Parent pointer for reconstruction
    parent = {start_state: None}

    # We want to find a path from depot to depot that visits some customers
    # and returns to depot. We'll search for the shortest such path that
    # visits at least one customer and is feasible.

    # To limit search, we'll use a modified Dijkstra that tracks
    # (node, battery, visited_set) and stops when we return to depot
    # with at least one customer visited.

    # Actually, let's think differently. We want to build a route that
    # visits a subset of remaining customers and returns to depot.
    # The route is a sequence of nodes. We can use Dijkstra where the
    # state is (current_node, battery, visited_customers_set).
    # The goal is to reach (0, any_battery, any_visited) where visited is non-empty.

    # But the state space is huge. Let's limit it.
    # We'll limit the number of customers per route to max_route_length.

    # Alternative: use a greedy approach with Dijkstra for the next customer.
    # Let's try a different approach: build the route incrementally.
    # At each step, from the current node, find the best next customer or station
    # using a priority queue.

    # Let me implement a simpler approach: iterative insertion with Dijkstra-like
    # selection of the next node to visit.

    # Actually, let's implement a proper Dijkstra on the state space.
    # State: (node, battery, visited_mask) where visited_mask is a bitmask of customers.
    # But 25 customers means 2^25 states, which is too many.

    # Let's use a different strategy: build the route by repeatedly selecting
    # the next customer that minimizes the incremental cost, using Dijkstra
    # to find the shortest path from current node to each candidate customer
    # (possibly via a station).

    # Simpler approach: use a greedy method with lookahead.
    # From the current node, for each unvisited customer, compute the cost
    # of visiting it next (including any necessary charging). Pick the best one.

    # Let's implement this greedy approach with Dijkstra for path finding.

    # State for building a route:
    current_node = 0
    current_battery = battery_max
    current_time = 0.0
    current_load = 0
    route = []
    visited = set()

    while True:
        if not remaining_customers - visited:
            break

        # Try to find the best next customer to visit
        best_cost = float('inf')
        best_customer = None
        best_route_to_customer = None

        for cust in remaining_customers - visited:
            # Find the shortest path from current_node to cust,
            # possibly via a station, such that:
            # 1. Battery doesn't go below 0
            # 2. Time window is satisfied
            # 3. Load doesn't exceed capacity

            # We can use Dijkstra on the graph where nodes are (depot, customers, stations)
            # and edges are direct travel. But we need to handle charging.

            # Simplified: try direct path, then via each station.
            candidates = [cust] + list(stations_set)

            # For each candidate intermediate node, check if we can reach it
            # and then reach cust from there.

            # Direct path
            d_direct = dist[current_node][cust]
            if current_battery - d_direct >= -1e-9 and current_load + demands[cust] <= capacity:
                t_arrive = current_time + d_direct / speed
                if t_arrive <= due[cust] + 1e-9:
                    # Check if we can return to depot after serving cust
                    d_back = dist[cust][0]
                    if current_battery - d_direct - d_back >= -1e-9:
                        cost = d_direct
                        if cost < best_cost:
                            best_cost = cost
                            best_customer = cust
                            best_route_to_customer = [cust]

            # Via each station
            for st in stations_set:
                d1 = dist[current_node][st]
                if current_battery - d1 >= -1e-9:
                    t1 = current_time + d1 / speed
                    # Charge at station
                    batt1 = max(0, current_battery - d1)
                    energy = battery_max - batt1
                    t1 += energy * recharge_rate
                    batt1 = battery_max

                    d2 = dist[st][cust]
                    if batt1 - d2 >= -1e-9 and current_load + demands[cust] <= capacity:
                        t2 = t1 + d2 / speed
                        if t2 <= due[cust] + 1e-9:
                            # Check if we can return to depot
                            d_back = dist[cust][0]
                            if batt1 - d2 - d_back >= -1e-9:
                                cost = d1 + d2
                                if cost < best_cost:
                                    best_cost = cost
                                    best_customer = cust
                                    best_route_to_customer = [st, cust]

        if best_customer is None:
            # Can't visit any more customers in this route
            break

        # Add the route to customer
        for node in best_route_to_customer:
            route.append(node)
            d = dist[current_node][node]
            current_battery -= d
            current_time += d / speed
            if node in stations_set:
                energy = battery_max - max(0, current_battery)
                current_time += energy * recharge_rate
                current_battery = battery_max
            else:
                current_load += demands[node]
                if current_time < ready[node]:
                    current_time = ready[node]
                current_time += service[node]
            visited.add(node)
            current_node = node

        # Check if we can still return to depot
        d_back = dist[current_node][0]
        if current_battery - d_back < -1e-9:
            # This shouldn't happen due to checks above, but just in case
            break

        # Limit route length
        if len([n for n in route if n not in stations_set]) >= max_route_length:
            break

    if not route:
        return None, float('inf')

    # Verify the route is feasible
    d_total, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity)
    if not valid:
        return None, float('inf')

    return route, d_total


def _greedy_construct(customers, dist, instance, n_customers, stations_set,
                      battery_max, speed, recharge_rate, horizon, capacity):
    """
    Construct routes using Dijkstra-based route building.
    """
    routes = []
    remaining = set(customers)

    while remaining:
        route, d = _dijkstra_route_build(
            remaining, dist, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity
        )
        if route is None:
            # Fallback: one customer per route
            for cust in remaining:
                routes.append([cust])
            break
        routes.append(route)
        for node in route:
            if node not in stations_set:
                remaining.discard(node)

    return routes


def _two_opt(route, dist, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity):
    """Apply 2-opt improvement to a single route."""
    n = len(route)
    if n < 3:
        return route

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                # Quick check: only consider if it might improve
                # Full feasibility check
                d_new, valid = _simulate_route(new_route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
                if valid:
                    d_old, _ = _simulate_route(route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
                    if d_new < d_old - 1e-9:
                        route = new_route
                        improved = True
    return route


def _local_search(routes, dist, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity,
                  time_limit=30.0):
    """Apply 2-opt and relocate moves."""
    start = time.time()
    improved = True

    while improved and (time.time() - start) < time_limit:
        improved = False

        # 2-opt within each route
        for ri in range(len(routes)):
            if (time.time() - start) > time_limit:
                return routes
            routes[ri] = _two_opt(
                routes[ri], dist, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity
            )

        # Relocate moves
        for ri in range(len(routes)):
            for pos in range(len(routes[ri])):
                if (time.time() - start) > time_limit:
                    return routes
                cust = routes[ri][pos]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos2 in range(len(routes[rj]) + 1):
                        new_ri = routes[ri][:pos] + routes[ri][pos + 1:]
                        new_rj = routes[rj][:pos2] + [cust] + routes[rj][pos2:]

                        # Check feasibility
                        valid_ri = True
                        valid_rj = True
                        d_ri = 0.0
                        d_rj = 0.0

                        if new_ri:
                            d_ri, valid_ri = _simulate_route(new_ri, dist, instance,
                                                             n_customers, stations_set,
                                                             battery_max, speed,
                                                             recharge_rate, horizon, capacity)
                        if new_rj:
                            d_rj, valid_rj = _simulate_route(new_rj, dist, instance,
                                                             n_customers, stations_set,
                                                             battery_max, speed,
                                                             recharge_rate, horizon, capacity)

                        if valid_ri and valid_rj:
                            old_total = _compute_total_distance([routes[ri], routes[rj]], dist)
                            new_total = d_ri + d_rj
                            if new_total < old_total - 1e-9:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

    return routes


def _simulated_annealing(routes, dist, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity,
                         time_limit=20.0):
    """Apply simulated annealing to improve routes."""
    start = time.time()
    current_routes = [list(r) for r in routes]
    current_dist = _compute_total_distance(current_routes, dist)
    best_routes = [list(r) for r in current_routes]
    best_dist = current_dist

    T = 50.0
    cooling = 0.995

    while (time.time() - start) < time_limit and T > 0.1:
        # Generate a random perturbation
        new_routes = [list(r) for r in current_routes]
        move_type = random.random()

        if move_type < 0.5:
            # 2-opt within a route
            ri = random.randint(0, len(new_routes) - 1)
            route = new_routes[ri]
            if len(route) >= 3:
                i, j = sorted(random.sample(range(len(route)), 2))
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d, valid = _simulate_route(new_route, dist, instance, n_customers,
                                           stations_set, battery_max, speed,
                                           recharge_rate, horizon, capacity)
                if valid:
                    new_routes[ri] = new_route
        else:
            # Relocate a customer
            ri = random.randint(0, len(new_routes) - 1)
            if len(new_routes[ri]) > 0:
                pos = random.randint(0, len(new_routes[ri]) - 1)
                cust = new_routes[ri][pos]
                rj = random.randint(0, len(new_routes) - 1)
                pos2 = random.randint(0, len(new_routes[rj]))
                new_ri = new_routes[ri][:pos] + new_routes[ri][pos + 1:]
                new_rj = new_routes[rj][:pos2] + [cust] + new_routes[rj][pos2:]

                valid_ri = True
                valid_rj = True
                d_ri = 0.0
                d_rj = 0.0
                if new_ri:
                    d_ri, valid_ri = _simulate_route(new_ri, dist, instance, n_customers,
                                                     stations_set, battery_max, speed,
                                                     recharge_rate, horizon, capacity)
                if new_rj:
                    d_rj, valid_rj = _simulate_route(new_rj, dist, instance, n_customers,
                                                     stations_set, battery_max, speed,
                                                     recharge_rate, horizon, capacity)

                if valid_ri and valid_rj:
                    new_routes[ri] = new_ri
                    new_routes[rj] = new_rj

        new_dist = _compute_total_distance(new_routes, dist)
        delta = new_dist - current_dist

        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_routes = [list(r) for r in current_routes]
                best_dist = current_dist

        T *= cooling

    return best_routes


def plan(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))

    # Build distance matrix
    dist = _build_dist_matrix(coords)

    # Deadline guard
    deadline = time.time() + 140.0

    # Phase 1: Dijkstra-based route construction
    routes = _greedy_construct(
        customers, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )

    # Validate
    valid, total_dist = _validate_solution(
        routes, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )
    if not valid:
        # Fallback: one customer per route
        routes = [[v] for v in customers]

    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        routes = _local_search(
            routes, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            time_limit=max(1.0, deadline - time.time() - 10.0)
        )

    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        routes = _simulated_annealing(
            routes, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            time_limit=max(1.0, deadline - time.time() - 5.0)
        )

    # Final validation
    valid, total_dist = _validate_solution(
        routes, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )
    if not valid:
        routes = [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
