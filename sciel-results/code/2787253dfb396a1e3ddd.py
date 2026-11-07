"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate_route(route, D, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate, horizon, stations_set):
    """
    Simulate a single route starting and ending at depot (node 0).
    Returns (feasible, total_distance, total_time, final_battery).
    """
    if not route:
        return True, 0.0, 0.0, battery_cap

    total_demand = sum(demands[v] for v in route)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    t = 0.0
    battery = battery_cap
    total_dist = 0.0
    prev = 0

    for v in route:
        d = D[prev, v]
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if v in stations_set:
            energy_added = battery_cap - battery
            t += energy_added * recharge_rate
            battery = battery_cap
        else:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, 0.0
            t += service[v]

        total_dist += d
        prev = v

    d = D[prev, 0]
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return False, 0.0, 0.0, 0.0
    total_dist += d

    if t > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0

    return True, total_dist, t, battery


def _route_distance(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

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

    stations_set = set(stations)
    customers = list(range(1, n_customers + 1))

    D = _build_distance_matrix(coords)

    # Phase 1: Greedy insertion construction
    customers_sorted = sorted(customers, key=lambda v: due[v])

    routes = []

    for cust in customers_sorted:
        best_pos = -1
        best_route_idx = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                feasible, new_dist, _, _ = _simulate_route(
                    new_route, D, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                if feasible:
                    old_dist = _route_distance(route, D)
                    cost_increase = new_dist - old_dist
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_pos = pos
                        best_route_idx = r_idx

        if best_route_idx != -1:
            route = routes[best_route_idx]
            new_route = route[:best_pos] + [cust] + route[best_pos:]
            routes[best_route_idx] = new_route
        else:
            routes.append([cust])

    # Phase 2: Local search with OR-opt and inter-route moves
    route_dists = [_route_distance(r, D) for r in routes]

    improved = True
    while improved and (time.time() < deadline - 3.0):
        improved = False

        # OR-opt: move 1, 2, or 3 consecutive customers within a route
        for r_idx, route in enumerate(routes):
            n = len(route)
            if n < 2:
                continue

            for seg_len in [1, 2, 3]:
                if n < seg_len + 1:
                    continue
                for i in range(n - seg_len + 1):
                    segment = route[i:i + seg_len]
                    remaining = route[:i] + route[i + seg_len:]
                    if not remaining:
                        continue
                    for j in range(len(remaining) + 1):
                        if j <= i and j >= i + seg_len:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        feasible, new_dist, _, _ = _simulate_route(
                            new_route, D, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                        if feasible and new_dist < route_dists[r_idx] - 1e-9:
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # Inter-route moves: move a customer from one route to another
        for src_idx in range(len(routes)):
            if not routes[src_idx]:
                continue
            route = routes[src_idx]
            for i in range(len(route)):
                cust = route[i]
                for dst_idx in range(len(routes)):
                    if dst_idx == src_idx:
                        continue
                    dst_route = routes[dst_idx]
                    for j in range(len(dst_route) + 1):
                        new_dst = dst_route[:j] + [cust] + dst_route[j:]
                        new_src = route[:i] + route[i + 1:]
                        feasible_dst, dist_dst, _, _ = _simulate_route(
                            new_dst, D, demands, ready, due, service,
                            capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                        if not feasible_dst:
                            continue
                        if new_src:
                            feasible_src, dist_src, _, _ = _simulate_route(
                                new_src, D, demands, ready, due, service,
                                capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                        else:
                            feasible_src, dist_src = True, 0.0
                        if not feasible_src:
                            continue
                        old_total = route_dists[src_idx] + route_dists[dst_idx]
                        new_total = dist_src + dist_dst
                        if new_total < old_total - 1e-9:
                            routes[src_idx] = new_src
                            routes[dst_idx] = new_dst
                            route_dists[src_idx] = dist_src
                            route_dists[dst_idx] = dist_dst
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

        # Try merging two routes if combined is feasible
        for i in range(len(routes)):
            if not routes[i]:
                continue
            for j in range(i + 1, len(routes)):
                if not routes[j]:
                    continue
                combined = routes[i] + routes[j]
                feasible, combined_dist, _, _ = _simulate_route(
                    combined, D, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                if feasible:
                    old_total = route_dists[i] + route_dists[j]
                    if combined_dist < old_total - 1e-9:
                        routes[i] = combined
                        routes[j] = []
                        route_dists[i] = combined_dist
                        route_dists[j] = 0.0
                        improved = True
                        break
            if improved:
                break

    # Clean up empty routes
    routes = [r for r in routes if r]

    # Final check: ensure all customers are served exactly once
    all_served = []
    for route in routes:
        all_served.extend(route)
    if sorted(all_served) != sorted(customers):
        return [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
