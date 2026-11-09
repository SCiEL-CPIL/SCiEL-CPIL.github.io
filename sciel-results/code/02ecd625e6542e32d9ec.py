"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time


def _dist2d(coords, i, j):
    dx = coords[i][0] - coords[j][0]
    dy = coords[i][1] - coords[j][1]
    return sqrt(dx * dx + dy * dy)


def _route_distance(coords, route):
    """Total Euclidean distance of a route starting and ending at depot (node 0)."""
    if not route:
        return 0.0
    d = _dist2d(coords, 0, route[0])
    for k in range(len(route) - 1):
        d += _dist2d(coords, route[k], route[k + 1])
    d += _dist2d(coords, route[-1], 0)
    return d


def _check_route_feasibility(coords, demands, ready, due, service,
                              battery, recharge_rate, horizon, route,
                              capacity, n_customers, stations):
    """
    Check if a single route is feasible.
    Returns (feasible, total_distance, end_time, end_battery).
    """
    if not route:
        return True, 0.0, 0.0, battery

    # Check capacity
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    # Simulate
    t = 0.0
    b = battery
    prev = 0
    for v in route:
        d = _dist2d(coords, prev, v)
        t += d
        b -= d
        if b < -1e-9:
            return False, 0.0, t, b
        if v <= n_customers:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, t, b
            t += service[v]
        # Stations have service 0, and we don't need time window checks
        prev = v

    # Return to depot
    d = _dist2d(coords, prev, 0)
    t += d
    b -= d
    if b < -1e-9:
        return False, 0.0, t, b
    if t > horizon + 1e-9:
        return False, 0.0, t, b

    return True, _route_distance(coords, route), t, b


def _insert_customer(route, coords, demands, ready, due, service,
                     battery, recharge_rate, horizon, capacity,
                     n_customers, stations, cust):
    """
    Try to insert customer 'cust' into 'route' at the best feasible position.
    Returns (new_route, improvement) if feasible, else (None, 0).
    improvement = old_dist - new_dist (positive if new is better).
    """
    old_dist = _route_distance(coords, route)

    # Check capacity
    current_demand = sum(demands[v] for v in route if v <= n_customers)
    if current_demand + demands[cust] > capacity:
        return None, 0.0

    best_pos = None
    best_new_dist = float('inf')

    for pos in range(len(route) + 1):
        new_route = route[:pos] + [cust] + route[pos:]
        # Quick check: just check if feasible
        feasible, nd, _, _ = _check_route_feasibility(
            coords, demands, ready, due, service,
            battery, recharge_rate, horizon, new_route,
            capacity, n_customers, stations
        )
        if feasible and nd < best_new_dist:
            best_new_dist = nd
            best_pos = pos

    if best_pos is None:
        return None, 0.0

    new_route = route[:best_pos] + [cust] + route[best_pos:]
    improvement = old_dist - best_new_dist
    return new_route, improvement


def _nearest_neighbor_route(coords, demands, ready, due, service,
                            battery, recharge_rate, horizon, capacity,
                            n_customers, stations, unassigned):
    """
    Build a route using nearest neighbor heuristic starting from depot.
    Returns the route (list of customer ids).
    """
    route = []
    remaining = list(unassigned)
    current = 0
    t = 0.0
    b = battery
    demand_so_far = 0

    while remaining:
        best_cust = None
        best_dist = float('inf')
        for c in remaining:
            d = _dist2d(coords, current, c)
            if d < best_dist:
                # Check if we can reach and serve
                t2 = t + d
                b2 = b - d
                if b2 < -1e-9:
                    continue
                if c <= n_customers:
                    if t2 < ready[c]:
                        t2 = ready[c]
                    if t2 > due[c] + 1e-9:
                        continue
                    t2 += service[c]
                    if demand_so_far + demands[c] > capacity:
                        continue
                best_dist = d
                best_cust = c

        if best_cust is None:
            break

        # Move to best_cust
        d = _dist2d(coords, current, best_cust)
        t += d
        b -= d
        if best_cust <= n_customers:
            if t < ready[best_cust]:
                t = ready[best_cust]
            t += service[best_cust]
            demand_so_far += demands[best_cust]
        route.append(best_cust)
        remaining.remove(best_cust)
        current = best_cust

    return route


def _solve_instance(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance.get("speed", 1.0)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]

    all_customers = list(range(1, n_customers + 1))

    # Initial solution: one customer per route
    routes = [[c] for c in all_customers]

    # Try to merge/insert customers into existing routes to reduce total distance
    # Sort customers by due time for better insertion order
    unassigned = sorted(all_customers, key=lambda c: due[c])

    # Start with empty routes and insert one by one
    routes = []
    for c in unassigned:
        best_route_idx = None
        best_improvement = -float('inf')
        best_new_routes = None

        # Try inserting into each existing route
        for i, route in enumerate(routes):
            new_route, imp = _insert_customer(
                route, coords, demands, ready, due, service,
                battery, recharge_rate, horizon, capacity,
                n_customers, stations, c
            )
            if new_route is not None and imp > best_improvement:
                best_improvement = imp
                best_route_idx = i
                best_new_routes = new_route

        if best_route_idx is not None:
            new_routes = routes[:best_route_idx] + [best_new_routes] + routes[best_route_idx + 1:]
            routes = new_routes
        else:
            routes.append([c])

    # Local improvement: try 2-opt within each route and reinsertion
    for _iter in range(5):  # Limited iterations
        improved = False
        # Try reinserting each customer into a different position/route
        for c in list(all_customers):
            # Find which route c is in
            route_idx = None
            for i, route in enumerate(routes):
                if c in route:
                    route_idx = i
                    break
            if route_idx is None:
                continue

            # Remove c from its route
            old_route = routes[route_idx]
            remaining_route = [v for v in old_route if v != c]

            best_target_idx = None
            best_improvement = -float('inf')
            best_new_routes = None

            for i, route in enumerate(routes):
                if i == route_idx:
                    # Try reinserting into same route at different position
                    new_route, imp = _insert_customer(
                        remaining_route, coords, demands, ready, due, service,
                        battery, recharge_rate, horizon, capacity,
                        n_customers, stations, c
                    )
                    if new_route is not None:
                        new_routes = routes[:i] + [new_route] + routes[i + 1:]
                        # Only accept if it improves
                        old_total = sum(_route_distance(coords, r) for r in routes)
                        new_total = sum(_route_distance(coords, r) for r in new_routes)
                        imp_actual = old_total - new_total
                        if imp_actual > best_improvement:
                            best_improvement = imp_actual
                            best_target_idx = i
                            best_new_routes = new_routes
                else:
                    # Try inserting into different route
                    new_route, imp = _insert_customer(
                        route, coords, demands, ready, due, service,
                        battery, recharge_rate, horizon, capacity,
                        n_customers, stations, c
                    )
                    if new_route is not None:
                        new_routes = routes[:i] + [new_route] + routes[i + 1:]
                        # Remove empty route if route_idx route becomes empty
                        if not remaining_route:
                            new_routes = [r for j, r in enumerate(new_routes) if j != route_idx or r]
                        old_total = sum(_route_distance(coords, r) for r in routes)
                        new_total = sum(_route_distance(coords, r) for r in new_routes)
                        imp_actual = old_total - new_total
                        if imp_actual > best_improvement:
                            best_improvement = imp_actual
                            best_target_idx = i
                            best_new_routes = new_routes

            if best_target_idx is not None and best_improvement > 1e-6:
                routes = best_new_routes
                improved = True

        if not improved:
            break

    # Compute total distance
    total_dist = sum(_route_distance(coords, r) for r in routes)

    # Verify all customers served
    served = set()
    for r in routes:
        for v in r:
            if v <= n_customers:
                served.add(v)
    if len(served) != n_customers:
        # Fallback: one per route
        routes = [[c] for c in all_customers]

    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, leave margin

    routes = _solve_instance(instance)

    # Verify feasibility
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance["horizon"]

    all_ok = True
    for r in routes:
        ok, _, _, _ = _check_route_feasibility(
            coords, demands, ready, due, service,
            battery, recharge_rate, horizon, r,
            capacity, n_customers, stations
        )
        if not ok:
            all_ok = False
            break

    if not all_ok:
        # Fallback to trivial solution
        customers = list(range(1, n_customers + 1))
        routes = [[c] for c in customers]

    return routes
# EVOLVE-BLOCK-END
