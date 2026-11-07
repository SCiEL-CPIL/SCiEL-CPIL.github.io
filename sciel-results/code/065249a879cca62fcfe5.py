"""EV routing with time windows and charging."""
# EVOLVE-BLOCK-START
import time
import math
import random
import heapq
from itertools import combinations

import numpy as np


def _euclid(coords, i, j):
    dx = coords[i, 0] - coords[j, 0]
    dy = coords[i, 1] - coords[j, 1]
    return math.sqrt(dx * dx + dy * dy)


def _build_dist(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclid(coords, i, j)
            D[i, j] = d
            D[j, i] = d
    return D


def _validate_route(route, instance, D):
    """Return (valid, total_distance)."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]

    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return False, float('inf')

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = D[prev, v]
        t += d / speed
        battery -= d
        total_dist += d
        if battery < -1e-9:
            return False, float('inf')

        if v in stations:
            # Charging time
            charge_time = max(0, (battery_max - battery) * recharge_rate)
            t += charge_time
            battery = battery_max
        else:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, float('inf')
            t += service[v]

        prev = v

    # Return to depot
    d = D[prev, 0]
    t += d / speed
    battery -= d
    total_dist += d
    if battery < -1e-9:
        return False, float('inf')
    if t > horizon + 1e-9:
        return False, float('inf')

    return True, total_dist


def _route_distance(route, D):
    total = 0.0
    prev = 0
    for v in route:
        total += D[prev, v]
        prev = v
    total += D[prev, 0]
    return total


def _savings_construct(instance, D, deadline):
    """Savings heuristic to build initial routes."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]

    # Start with each customer as its own route
    routes = [[c] for c in customers]
    route_of = {c: i for i, c in enumerate(customers)}
    route_load = [demands[c] for c in customers]

    # Compute savings: saving of merging route i and route j
    # saving = D[0][a] + D[0][b] - D[a][b] where a is last of route i, b is first of route j
    # For simplicity, try all pairs and pick best feasible merge

    changed = True
    while changed:
        changed = False
        best_saving = 0
        best_pair = None
        best_pos = None  # (route_idx, insert_pos)

        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                if route_load[i] + route_load[j] > capacity:
                    continue
                # Try inserting route j into route i at various positions
                for pos in range(len(routes[i]) + 1):
                    # Compute distance change
                    route_i = routes[i]
                    if pos == 0:
                        prev_node = 0
                        next_node = route_i[0]
                    elif pos == len(route_i):
                        prev_node = route_i[-1]
                        next_node = 0
                    else:
                        prev_node = route_i[pos - 1]
                        next_node = route_i[pos]

                    # Distance of current route i
                    d_old = _route_distance(route_i, D)
                    # New route
                    new_route = route_i[:pos] + routes[j] + route_i[pos:]
                    d_new = _route_distance(new_route, D)
                    saving = d_old - d_new
                    if saving > best_saving + 1e-9:
                        # Check validity
                        valid, _ = _validate_route(new_route, instance, D)
                        if valid:
                            best_saving = saving
                            best_pair = (i, j)
                            best_pos = pos

        if best_pair is not None:
            i, j = best_pair
            new_route = routes[i][:best_pos] + routes[j] + routes[i][best_pos:]
            routes[i] = new_route
            routes.pop(j)
            route_load[i] += route_load[j]
            route_load.pop(j)
            # Update route_of
            for v in new_route:
                route_of[v] = i
            changed = True

        if time.time() > deadline - 10:
            break

    return routes


def _two_opt(route, D, instance, max_iters=500):
    """2-opt improvement on a single route."""
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d_old = _route_distance(route, D)
                d_new = _route_distance(new_route, D)
                if d_new < d_old - 1e-9:
                    valid, _ = _validate_route(new_route, instance, D)
                    if valid:
                        route = new_route
                        improved = True
    return route


def _or_opt(route, D, instance, max_iters=200):
    """Or-opt: move 1-3 consecutive nodes to another position."""
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(route)
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                seg = route[i:i + seg_len]
                for j in range(n - seg_len + 1):
                    if j == i or (j > i and j < i + seg_len):
                        continue
                    # Remove seg and insert at position j
                    new_route = route[:i] + route[i + seg_len:]
                    # Adjust j if removal shifted positions
                    insert_j = j
                    if j > i:
                        insert_j = j - seg_len
                    new_route = new_route[:insert_j] + seg + new_route[insert_j:]
                    if new_route == route:
                        continue
                    d_old = _route_distance(route, D)
                    d_new = _route_distance(new_route, D)
                    if d_new < d_old - 1e-9:
                        valid, _ = _validate_route(new_route, instance, D)
                        if valid:
                            route = new_route
                            improved = True
    return route


def _lns(instance, D, start_routes, deadline, rng):
    """Large Neighborhood Search."""
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    demands = instance["demands"]

    best_routes = [list(r) for r in start_routes]
    best_dist = sum(_route_distance(r, D) for r in best_routes)
    current_routes = [list(r) for r in start_routes]
    current_dist = best_dist

    all_customers = set(range(1, n_customers + 1))

    while time.time() < deadline - 5:
        # Randomly remove some customers
        n_remove = rng.randint(3, 8)
        removed = set()
        for _ in range(n_remove):
            c = rng.choice(list(all_customers - removed))
            removed.add(c)

        # Remove from current routes
        new_routes = []
        for route in current_routes:
            new_route = [v for v in route if v not in removed]
            if new_route:
                new_routes.append(new_route)

        # Insert removed customers one by one into best feasible position
        for c in removed:
            best_pos = None
            best_cost = float('inf')
            for r_idx, route in enumerate(new_routes):
                for pos in range(len(route) + 1):
                    candidate = route[:pos] + [c] + route[pos:]
                    # Check capacity
                    load = sum(demands[v] for v in candidate)
                    if load > capacity:
                        continue
                    valid, _ = _validate_route(candidate, instance, D)
                    if not valid:
                        continue
                    cost = _route_distance(candidate, D)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (r_idx, pos)

            if best_pos is None:
                # Can't insert, revert
                current_routes = [list(r) for r in best_routes]
                current_dist = best_dist
                break
            else:
                r_idx, pos = best_pos
                new_routes[r_idx] = new_routes[r_idx][:pos] + [c] + new_routes[r_idx][pos:]
        else:
            # All inserted
            current_dist = sum(_route_distance(r, D) for r in new_routes)
            if current_dist < best_dist:
                best_routes = [list(r) for r in new_routes]
                best_dist = current_dist
            current_routes = [list(r) for r in new_routes]

    return best_routes, best_dist


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    D = _build_dist(coords)
    n_customers = instance["n_customers"]

    rng = random.Random(42)

    # Initial construction via savings
    routes = _savings_construct(instance, D, deadline)

    # Local improvement
    for i in range(len(routes)):
        routes[i] = _two_opt(routes[i], D, instance)
        routes[i] = _or_opt(routes[i], D, instance)

    current_dist = sum(_route_distance(r, D) for r in routes)

    # LNS
    if time.time() < deadline - 15:
        routes, current_dist = _lns(instance, D, routes, deadline, rng)

    # Final 2-opt pass
    if time.time() < deadline - 5:
        for i in range(len(routes)):
            routes[i] = _two_opt(routes[i], D, instance, max_iters=200)

    # Validate final solution
    total_dist = 0.0
    for route in routes:
        valid, d = _validate_route(route, instance, D)
        if not valid:
            # Fallback: one customer per route
            customers = list(range(1, n_customers + 1))
            return [[v] for v in customers]
        total_dist += d

    return routes
# EVOLVE-BLOCK-END
