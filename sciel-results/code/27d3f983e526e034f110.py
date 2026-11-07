"""EV routing with time windows: constructive + local search with or-opt."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 15.0  # seconds before hard limit

def _check_deadline():
    if _DEADLINE is not None and time.time() > _DEADLINE - _DEADLINE_MARGIN:
        return True
    return False


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _route_distance(route, dist):
    """Total Euclidean distance including depot legs."""
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def _simulate_route(route, instance, dist):
    """
    Simulate a route. Returns (feasible, total_distance, final_time, final_battery).
    If infeasible, returns (False, inf, inf, 0).
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = instance["_station_set"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    total_demand = sum(demands[v] for v in route if v not in stations)
    if total_demand > capacity:
        return (False, float('inf'), float('inf'), 0.0)

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = dist[prev][v]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return (False, float('inf'), float('inf'), 0.0)

        if v in stations:
            # Recharge: time = (battery_max - battery) * recharge_rate
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = battery_max
        else:
            # Customer: time window
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return (False, float('inf'), float('inf'), 0.0)
            t += service[v]

        prev = v

    # Return to depot
    d = dist[prev][0]
    t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return (False, float('inf'), float('inf'), 0.0)
    if t > horizon + 1e-9:
        return (False, float('inf'), float('inf'), 0.0)

    return (True, total_dist, t, battery)


def _solve_instance(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    due = instance["due"]
    ready = instance["ready"]
    service = instance["service"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    # Precompute distances
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    # Store station set for faster lookup
    instance["_station_set"] = set(stations)

    # Sort customers by due time (earliest due first)
    customers_sorted = sorted(customers, key=lambda v: (due[v], ready[v]))

    # Greedy insertion: build routes
    max_routes = 10
    routes = [[] for _ in range(max_routes)]
    route_loads = [0.0] * max_routes

    # Start with first customer in route 0
    if customers_sorted:
        routes[0].append(customers_sorted[0])
        route_loads[0] += demands[customers_sorted[0]]
        unassigned = set(customers_sorted[1:])
    else:
        unassigned = set()

    for cust in customers_sorted:
        if cust not in unassigned:
            continue
        if _check_deadline():
            break
        best_route = -1
        best_cost = float('inf')
        best_pos = -1
        for r in range(len(routes)):
            if route_loads[r] + demands[cust] > capacity:
                continue
            route = routes[r]
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, d, t, b = _simulate_route(new_route, instance, dist)
                if feasible and d < best_cost:
                    best_cost = d
                    best_route = r
                    best_pos = pos
        if best_route >= 0:
            routes[best_route].insert(best_pos, cust)
            route_loads[best_route] += demands[cust]
            unassigned.discard(cust)
        else:
            # Open a new route
            new_r = len(routes)
            if new_r < max_routes:
                routes.append([cust])
                route_loads.append(demands[cust])
                unassigned.discard(cust)
            else:
                # Force into some route (shouldn't happen)
                routes[0].append(cust)
                route_loads[0] += demands[cust]
                unassigned.discard(cust)

    # Remove empty routes
    routes = [r for r in routes if r]

    # Local search: 2-opt, or-opt, relocate, and swap
    improved = True
    iterations = 0
    max_iters = 300
    while improved and iterations < max_iters:
        improved = False
        iterations += 1
        if _check_deadline():
            break

        # 2-opt within each route
        for r in range(len(routes)):
            if len(routes[r]) < 2:
                continue
            route = routes[r]
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    f1, d1, t1, b1 = _simulate_route(route, instance, dist)
                    f2, d2, t2, b2 = _simulate_route(new_route, instance, dist)
                    if f2 and (not f1 or d2 < d1 - 1e-9):
                        routes[r] = new_route
                        route = new_route
                        improved = True

        # Or-opt: move 1 or 2 consecutive customers within a route
        for r in range(len(routes)):
            if len(routes[r]) < 3:
                continue
            route = routes[r]
            for seg_len in [1, 2]:
                for i in range(len(route) - seg_len + 1):
                    segment = route[i:i + seg_len]
                    # Try inserting segment at other positions
                    for j in range(len(route) - seg_len + 1):
                        if j == i or (seg_len == 2 and j == i + 1) or (seg_len == 1 and j in [i, i + 1]):
                            continue
                        # Remove segment
                        new_route = route[:i] + route[i + seg_len:]
                        # Insert at position j (adjust index if j > i)
                        insert_pos = j if j < i else j
                        new_route = new_route[:insert_pos] + segment + new_route[insert_pos:]
                        f1, d1, t1, b1 = _simulate_route(route, instance, dist)
                        f2, d2, t2, b2 = _simulate_route(new_route, instance, dist)
                        if f2 and (not f1 or d2 < d1 - 1e-9):
                            routes[r] = new_route
                            route = new_route
                            improved = True

        # Relocate between routes
        for r1 in range(len(routes)):
            if not routes[r1]:
                continue
            for i in range(len(routes[r1])):
                cust = routes[r1][i]
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2]) + 1):
                        new_r1 = routes[r1][:i] + routes[r1][i + 1:]
                        new_r2 = routes[r2][:j] + [cust] + routes[r2][j:]
                        f1, d1, t1, b1 = _simulate_route(new_r1, instance, dist)
                        f2, d2, t2, b2 = _simulate_route(new_r2, instance, dist)
                        if f2 and f1:
                            old_d = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                            new_d = _route_distance(new_r1, dist) + _route_distance(new_r2, dist)
                            if new_d < old_d - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Swap between routes
        for r1 in range(len(routes)):
            if len(routes[r1]) < 1:
                continue
            for i in range(len(routes[r1])):
                cust1 = routes[r1][i]
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for j in range(len(routes[r2])):
                        cust2 = routes[r2][j]
                        if cust1 == cust2:
                            continue
                        new_r1 = routes[r1][:]
                        new_r2 = routes[r2][:]
                        new_r1[i] = cust2
                        new_r2[j] = cust1
                        f1, d1, t1, b1 = _simulate_route(new_r1, instance, dist)
                        f2, d2, t2, b2 = _simulate_route(new_r2, instance, dist)
                        if f1 and f2:
                            old_d = _route_distance(routes[r1], dist) + _route_distance(routes[r2], dist)
                            new_d = _route_distance(new_r1, dist) + _route_distance(new_r2, dist)
                            if new_d < old_d - 1e-9:
                                routes[r1] = new_r1
                                routes[r2] = new_r2
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    # Remove empty routes again
    routes = [r for r in routes if r]

    return routes


def plan(instance):
    global _DEADLINE
    _DEADLINE = time.time() + 150.0 - 15.0  # 135s effective

    # Solve single instance
    routes = _solve_instance(instance)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            if v > 0:
                served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        # Fallback: one customer per route
        routes = [[v] for v in range(1, instance["n_customers"] + 1)]

    return routes
# EVOLVE-BLOCK-END
