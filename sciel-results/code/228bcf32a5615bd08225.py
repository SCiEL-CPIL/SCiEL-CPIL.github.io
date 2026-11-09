"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from itertools import combinations

# Global deadline guard
_T0 = time.time()
_DEADLINE = _T0 + 135.0  # safety margin before 150s kill


def _remaining_time():
    return _DEADLINE - time.time()


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_dist_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            d[j, i] = d[i, j]
    return d


def _simulate_route(route, instance, dist, battery, capacity, speed, recharge_rate, horizon):
    """
    Simulate a route. Returns (total_distance, valid).
    route: list of node ids (customers and stations), no depot.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_dist = 0.0
    curr_batt = battery
    curr_time = 0.0
    curr_load = 0.0
    prev_node = 0  # depot

    for node in route:
        d = dist[prev_node][node]
        curr_time += d / speed
        curr_batt -= d

        if curr_batt < -1e-9:
            return total_dist, False

        if node in instance["stations"]:
            # Charging station: add recharge time, refill battery
            energy_needed = battery - curr_batt
            curr_time += energy_needed * recharge_rate
            curr_batt = battery
        else:
            # Customer
            curr_time = max(curr_time, ready[node])
            if curr_time > due[node] + 1e-9:
                return total_dist, False
            curr_load += demands[node]
            if curr_load > capacity + 1e-9:
                return total_dist, False
            curr_time += service[node]

        total_dist += d
        prev_node = node

    # Return to depot
    d = dist[prev_node][0]
    curr_time += d / speed
    curr_batt -= d
    if curr_batt < -1e-9:
        return total_dist, False
    if curr_time > horizon + 1e-9:
        return total_dist, False

    total_dist += d
    return total_dist, True


def _construct_greedy_route(instance, dist, unvisited, battery, capacity, speed, recharge_rate, horizon, max_nodes=10):
    """
    Greedily build a feasible route starting from depot, visiting nodes from unvisited.
    Returns (route, visited_set) or (None, None) if no feasible node can be added.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    route = []
    visited = set()
    curr_batt = battery
    curr_time = 0.0
    curr_load = 0.0
    prev_node = 0

    for _ in range(max_nodes):
        best_node = None
        best_score = float('inf')

        candidates = list(unvisited - visited)
        # Also consider adding a station if battery is low
        if curr_batt < battery * 0.3:
            for s in stations:
                if s not in visited:
                    candidates.append(s)

        for node in candidates:
            d = dist[prev_node][node]
            new_time = curr_time + d / speed
            new_batt = curr_batt - d

            if new_batt < -1e-9:
                continue

            if node in stations:
                new_time += (battery - new_batt) * recharge_rate
                new_batt = battery
                new_load = curr_load
            else:
                new_time = max(new_time, ready[node])
                if new_time > due[node] + 1e-9:
                    continue
                new_load = curr_load + demands[node]
                if new_load > capacity + 1e-9:
                    continue
                new_time += service[node]

            # Check if we can return to depot
            d_back = dist[node][0]
            final_batt = new_batt - d_back
            final_time = new_time + d_back / speed
            if final_batt < -1e-9 or final_time > horizon + 1e-9:
                continue

            score = d
            if score < best_score:
                best_score = score
                best_node = node

        if best_node is None:
            break

        d = dist[prev_node][best_node]
        curr_time += d / speed
        curr_batt -= d

        if best_node in stations:
            curr_time += (battery - curr_batt) * recharge_rate
            curr_batt = battery
        else:
            curr_time = max(curr_time, ready[best_node])
            curr_time += service[best_node]
            curr_load += demands[best_node]
            visited.add(best_node)

        route.append(best_node)
        prev_node = best_node

    if not route:
        return None, None
    return route, visited


def _local_search(instance, dist, routes, battery, capacity, speed, recharge_rate, horizon, max_iter=500):
    """
    Improve routes using 2-opt and relocate moves within and across routes.
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    total_dist = sum(_simulate_route(r, instance, dist, battery, capacity, speed, recharge_rate, horizon)[0] for r in routes if r)

    for _ in range(max_iter):
        if _remaining_time() < 5.0:
            break

        improved = False

        # 2-opt within each route
        for ri, route in enumerate(routes):
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    d_new, valid = _simulate_route(new_route, instance, dist, battery, capacity, speed, recharge_rate, horizon)
                    d_old, _ = _simulate_route(route, instance, dist, battery, capacity, speed, recharge_rate, horizon)
                    if valid and d_new < d_old - 1e-6:
                        routes[ri] = new_route
                        total_dist = sum(_simulate_route(r, instance, dist, battery, capacity, speed, recharge_rate, horizon)[0] for r in routes if r)
                        improved = True
                        break
                if improved:
                    break

        # Relocate: move a customer from one route to another
        if improved:
            continue

        for ri in range(len(routes)):
            for si in range(len(routes[ri])):
                node = routes[ri][si]
                if node in stations:
                    continue
                # Try removing from ri
                new_ri = routes[ri][:si] + routes[ri][si + 1:]
                d_ri, valid_ri = _simulate_route(new_ri, instance, dist, battery, capacity, speed, recharge_rate, horizon) if new_ri else (0, True)
                if not valid_ri:
                    continue

                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos in range(len(routes[rj]) + 1):
                        new_rj = routes[rj][:pos] + [node] + routes[rj][pos:]
                        d_rj, valid_rj = _simulate_route(new_rj, instance, dist, battery, capacity, speed, recharge_rate, horizon)
                        if not valid_rj:
                            continue
                        total_new = (d_ri if new_ri else 0) + d_rj
                        total_old = sum(_simulate_route(r, instance, dist, battery, capacity, speed, recharge_rate, horizon)[0] for r in routes if r)
                        if total_new < total_old - 1e-6:
                            routes[ri] = new_ri
                            routes[rj] = new_rj
                            total_dist = total_new
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

    return routes


def plan(instance):
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 135.0

    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    dist = _build_dist_matrix(coords)
    customers = list(range(1, n_customers + 1))

    # Phase 1: Construct initial solution using greedy route building
    unvisited = set(customers)
    routes = []
    max_nodes_per_route = 8

    while unvisited:
        if _remaining_time() < 10.0:
            break
        route, visited = _construct_greedy_route(
            instance, dist, unvisited, battery, capacity, speed, recharge_rate, horizon, max_nodes_per_route
        )
        if route is None:
            # Force add a single customer
            c = min(unvisited, key=lambda c: due[c])
            routes.append([c])
            unvisited.discard(c)
        else:
            routes.append(route)
            unvisited -= visited

    # Ensure all customers are covered
    covered = set()
    for r in routes:
        for n in r:
            if n not in set(stations):
                covered.add(n)
    for c in customers:
        if c not in covered:
            routes.append([c])

    # Phase 2: Local search improvement
    if _remaining_time() > 15.0:
        routes = _local_search(instance, dist, routes, battery, capacity, speed, recharge_rate, horizon, max_iter=300)

    # Filter out empty routes
    routes = [r for r in routes if r]

    return routes
# EVOLVE-BLOCK-END
