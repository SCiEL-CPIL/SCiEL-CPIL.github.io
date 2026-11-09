"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random
from itertools import combinations


def _euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dist_matrix(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            d[j, i] = d[i, j]
    return d


def _check_route(route, instance, dist, nodes):
    """
    Check if a route is feasible. Returns (feasible, total_distance, end_time, end_battery)
    """
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    total_demand = sum(demands[v] for v in route)
    if total_demand > capacity:
        return False, 0, 0, 0

    t = 0.0
    battery = float(battery_cap)
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = dist[prev][v]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return False, 0, 0, 0
        t += d / speed

        if v in stations:
            energy_needed = battery_cap - battery
            recharge_time = energy_needed * recharge_rate
            t += recharge_time
            battery = float(battery_cap)
        else:
            # customer
            if t < ready[v]:
                t = float(ready[v])
            if t > due[v] + 1e-9:
                return False, 0, 0, 0
            t += service[v]

        prev = v

    # return to depot
    d = dist[prev][0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return False, 0, 0, 0
    t += d / speed
    if t > horizon + 1e-9:
        return False, 0, 0, 0

    return True, total_dist, t, battery


def _route_distance(route, dist):
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i + 1]]
    total += dist[route[-1]][0]
    return total


def _construct_initial(instance, dist):
    """
    Construct initial feasible solution using nearest neighbor with time windows.
    """
    customers = list(range(1, instance["n_customers"] + 1))
    random.shuffle(customers)

    routes = []
    unvisited = set(customers)

    while unvisited:
        # Start a new route from depot
        route = []
        prev = 0
        t = 0.0
        battery = float(instance["battery"])
        total_demand = 0
        total_dist = 0.0

        while unvisited:
            # Find nearest feasible customer
            best_node = -1
            best_dist = float('inf')
            for v in unvisited:
                d = dist[prev][v]
                if total_demand + instance["demands"][v] > instance["capacity"]:
                    continue
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                new_t = t + d / instance["speed"]
                if new_t < instance["ready"][v]:
                    new_t = float(instance["ready"][v])
                if new_t > instance["due"][v] + 1e-9:
                    continue
                # Check we can return to depot
                ret_d = dist[v][0]
                if new_battery - ret_d < -1e-9:
                    continue
                if new_t + instance["service"][v] + ret_d / instance["speed"] > instance["horizon"] + 1e-9:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_node = v

            if best_node == -1:
                break

            route.append(best_node)
            total_demand += instance["demands"][best_node]
            total_dist += best_dist
            battery -= best_dist
            t += best_dist / instance["speed"]
            if t < instance["ready"][best_node]:
                t = float(instance["ready"][best_node])
            t += instance["service"][best_node]
            unvisited.remove(best_node)
            prev = best_node

        if route:
            # Verify feasibility
            feasible, _, _, _ = _check_route(route, instance, dist, None)
            if feasible:
                routes.append(route)
            else:
                # If not feasible, add one by one
                for v in route:
                    if v in unvisited:
                        continue
                    # This shouldn't happen, but just in case
                    pass
                # Re-add and try individually
                for v in route:
                    unvisited.add(v)
                # Try to add each individually
                for v in sorted(route, key=lambda x: dist[0][x]):
                    if v in unvisited:
                        new_route = [v]
                        feasible, _, _, _ = _check_route(new_route, instance, dist, None)
                        if feasible:
                            routes.append(new_route)
                            unvisited.remove(v)
        else:
            # No customer could be added, force add nearest
            v = min(unvisited, key=lambda x: dist[0][x])
            routes.append([v])
            unvisited.remove(v)

    return routes


def _two_opt(route, dist, instance, max_iter=100):
    """
    2-opt improvement for a single route.
    """
    if len(route) < 2:
        return route

    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(len(route)):
            for j in range(i + 1, len(route)):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                feasible, _, _, _ = _check_route(new_route, instance, dist, None)
                if feasible:
                    old_d = _route_distance(route, dist)
                    new_d = _route_distance(new_route, dist)
                    if new_d < old_d - 1e-9:
                        route = new_route
                        improved = True
    return route


def _or_opt(route, dist, instance, max_iter=50):
    """
    Or-opt: move 1 or 2 consecutive customers to a different position.
    """
    if len(route) < 2:
        return route

    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for length in [1, 2]:
            for i in range(len(route) - length + 1):
                segment = route[i:i + length]
                rest = route[:i] + route[i + length:]
                for j in range(len(rest) + 1):
                    if i <= j < i + length:
                        continue
                    new_route = rest[:j] + segment + rest[j:]
                    if new_route == route:
                        continue
                    feasible, _, _, _ = _check_route(new_route, instance, dist, None)
                    if feasible:
                        old_d = _route_distance(route, dist)
                        new_d = _route_distance(new_route, dist)
                        if new_d < old_d - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route


def _try_insert_customer(route, customer, instance, dist):
    """
    Try to insert a customer into a route at the best position.
    Returns (success, new_route, additional_distance)
    """
    if not route:
        # Check if single customer route is feasible
        feasible, d, _, _ = _check_route([customer], instance, dist, None)
        if feasible:
            return True, [customer], d
        return False, route, 0

    best_pos = -1
    best_add = float('inf')
    best_route = None

    for pos in range(len(route) + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        feasible, _, _, _ = _check_route(new_route, instance, dist, None)
        if feasible:
            # Calculate additional distance
            if pos == 0:
                add = dist[0][customer] + dist[customer][route[0]] - dist[0][route[0]]
            elif pos == len(route):
                add = dist[route[-1]][customer] + dist[customer][0] - dist[route[-1]][0]
            else:
                add = dist[route[pos - 1]][customer] + dist[customer][route[pos]] - dist[route[pos - 1]][route[pos]]
            if add < best_add:
                best_add = add
                best_pos = pos
                best_route = new_route

    if best_route is not None:
        return True, best_route, best_add
    return False, route, 0


def _solve_instance(instance, deadline):
    """
    Solve a single instance within the deadline.
    """
    coords = instance["coords"]
    dist = _dist_matrix(coords)
    n_customers = instance["n_customers"]

    # Construct initial solution
    best_routes = _construct_initial(instance, dist)
    best_dist = sum(_route_distance(r, dist) for r in best_routes)

    start_time = time.time()
    elapsed = start_time - start_time

    # Local search: 2-opt and Or-opt on each route
    improved = True
    while improved and time.time() - start_time < deadline * 0.6:
        improved = False
        for i, route in enumerate(best_routes):
            if len(route) < 2:
                continue
            # 2-opt
            new_route = _two_opt(route, dist, instance, max_iter=50)
            if new_route != route:
                best_routes[i] = new_route
                best_dist = sum(_route_distance(r, dist) for r in best_routes)
                improved = True
            # Or-opt
            new_route = _or_opt(best_routes[i], dist, instance, max_iter=30)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                best_dist = sum(_route_distance(r, dist) for r in best_routes)
                improved = True

    # LNS: Remove and reinsert customers
    if time.time() - start_time < deadline * 0.85:
        for _ in range(50):
            if time.time() - start_time > deadline * 0.85:
                break

            # Remove 3-5 random customers
            n_remove = min(3, sum(len(r) for r in best_routes))
            removed = []
            for _ in range(n_remove):
                if not best_routes or all(len(r) == 0 for r in best_routes):
                    break
                # Pick a random customer from a random route
                non_empty = [i for i, r in enumerate(best_routes) if len(r) > 0]
                if not non_empty:
                    break
                ri = random.choice(non_empty)
                ci = random.randrange(len(best_routes[ri]))
                v = best_routes[ri].pop(ci)
                removed.append(v)
                if len(best_routes[ri]) == 0:
                    best_routes.pop(ri)

            # Reinsert removed customers
            for v in removed:
                inserted = False
                # Try to insert into existing routes
                for i in range(len(best_routes)):
                    success, new_route, _ = _try_insert_customer(best_routes[i], v, instance, dist)
                    if success:
                        best_routes[i] = new_route
                        inserted = True
                        break
                if not inserted:
                    # Create new route
                    best_routes.append([v])

            # Local search on affected routes
            for i, route in enumerate(best_routes):
                if len(route) >= 2:
                    best_routes[i] = _two_opt(route, dist, instance, max_iter=20)
                    best_routes[i] = _or_opt(best_routes[i], dist, instance, max_iter=10)

    return best_routes


def plan(instance):
    """
    Plan routes for an EV routing instance.
    """
    start_time = time.time()
    deadline = 40.0  # Leave margin for 3 instances

    # Precompute distance matrix
    coords = instance["coords"]
    dist = _dist_matrix(coords)

    # Solve
    routes = _solve_instance(instance, deadline)

    # Verify all customers are served
    served = set()
    for r in routes:
        for v in r:
            served.add(v)
    expected = set(range(1, instance["n_customers"] + 1))
    if served != expected:
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
