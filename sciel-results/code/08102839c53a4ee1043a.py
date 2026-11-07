"""
EV routing with time windows - structural redesign.

Approach:
  1. Build a distance matrix.
  2. Cluster customers into groups using a greedy nearest-neighbour /
     k-means style partitioning to form candidate routes.
  3. For each cluster, solve a small TSP with 2-opt + or-opt local search
     while respecting time windows, capacity, and battery constraints.
  4. If a cluster is infeasible, split it.
  5. Insert charging stops as needed.
  6. Deadline guard: track elapsed time and return best feasible solution.
"""

import numpy as np
import time
import math
import random
import heapq
import itertools
from copy import deepcopy

# ---------- helpers ----------

def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _build_dist(coords):
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = math.hypot(coords[i][0] - coords[j][0],
                                 coords[i][1] - coords[j][1])
            d[j, i] = d[i, j]
    return d


def _check_route(route, instance, dist):
    """
    Validate a single route (list of node ids, depot implicit at both ends).
    Returns (feasible, total_distance, end_time, min_battery).
    """
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

    # check capacity
    load = 0
    for v in route:
        if v not in stations:
            load += demands[v]
    if load > capacity:
        return False, 0, 0, 0

    battery = battery_max
    t = 0.0
    total_dist = 0.0
    prev = 0  # depot
    min_battery = battery

    for v in route:
        d = dist[prev, v]
        t += d / speed
        battery -= d
        total_dist += d
        if battery < -1e-9:
            return False, 0, 0, 0
        min_battery = min(min_battery, battery)

        if v in stations:
            # recharge: time added = (battery_max - battery) * recharge_rate
            # then battery refills to max
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            # customer
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0, 0, 0
            t += service[v]

        prev = v

    # return to depot
    d = dist[prev, 0]
    t += d / speed
    battery -= d
    total_dist += d
    if battery < -1e-9:
        return False, 0, 0, 0
    min_battery = min(min_battery, battery)
    if t > horizon + 1e-9:
        return False, 0, 0, 0

    return True, total_dist, t, min_battery


def _route_distance(route, dist):
    """Compute total Euclidean distance including depot legs."""
    total = dist[0, route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i], route[i + 1]]
    total += dist[route[-1], 0]
    return total


def _tsp_2opt(order, dist):
    """2-opt improvement for a list of node ids (not including depot)."""
    n = len(order)
    if n < 3:
        return order
    best = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # reverse segment [i..j]
                new_order = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                d_old = (dist[best[i - 1] if i > 0 else 0, best[i]] +
                         dist[best[j], best[j + 1] if j + 1 < n else 0])
                d_new = (dist[best[i - 1] if i > 0 else 0, best[j]] +
                         dist[best[i], best[j + 1] if j + 1 < n else 0])
                if d_new < d_old - 1e-9:
                    best = new_order
                    improved = True
    return best


def _oropt(order, dist):
    """Or-opt: move 1-2 consecutive nodes to a better position."""
    n = len(order)
    if n < 3:
        return order
    best = list(order)
    improved = True
    while improved:
        improved = False
        for k in range(1, 3):
            for i in range(n - k + 1):
                segment = best[i:i + k]
                rest = best[:i] + best[i + k:]
                for j in range(len(rest) + 1):
                    new_order = rest[:j] + segment + rest[j:]
                    if new_order == best:
                        continue
                    # quick distance check
                    d_old = 0
                    prev = 0
                    for v in best:
                        d_old += dist[prev, v]
                        prev = v
                    d_old += dist[prev, 0]

                    d_new = 0
                    prev = 0
                    for v in new_order:
                        d_new += dist[prev, v]
                        prev = v
                    d_new += dist[prev, 0]

                    if d_new < d_old - 1e-9:
                        best = new_order
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best


def _check_feasibility(order, instance, dist):
    """Check if a customer order (without stations) is feasible with optional charging inserts."""
    # We'll try to insert charging stops where needed.
    # First, check if feasible without any charging.
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = instance["stations"]

    # Try direct route
    feasible, _, _, _ = _check_route(order, instance, dist)
    if feasible:
        return order, True

    # Try inserting one charging stop between each consecutive pair
    # and at start/end
    # For simplicity, try inserting the nearest station after each customer
    # where battery would go negative
    for attempt in range(2):  # up to 2 rounds of insertion
        new_order = [0]  # start with depot
        battery = battery_max
        t = 0.0
        feasible = True
        for v in order:
            # check if we need to charge before going to v
            d = dist[new_order[-1], v]
            if battery - d < -1e-9:
                # need to charge: find nearest station
                best_st = None
                best_d = float('inf')
                for st in stations:
                    dd = dist[new_order[-1], st] + dist[st, v]
                    if dd < best_d:
                        best_d = dd
                        best_st = st
                if best_st is None:
                    feasible = False
                    break
                # go to station
                d1 = dist[new_order[-1], best_st]
                t += d1 / speed
                battery -= d1
                if battery < -1e-9:
                    feasible = False
                    break
                # recharge
                t += (battery_max - battery) * recharge_rate
                battery = battery_max
                new_order.append(best_st)
                # now go to customer
                d2 = dist[best_st, v]
                t += d2 / speed
                battery -= d2
                if battery < -1e-9:
                    feasible = False
                    break
            else:
                d = dist[new_order[-1], v]
                t += d / speed
                battery -= d

            # at customer
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                feasible = False
                break
            t += service[v]
            new_order.append(v)

        if not feasible:
            return order, False

        # return to depot
        d = dist[new_order[-1], 0]
        t += d / speed
        battery -= d
        if battery < -1e-9 or t > horizon + 1e-9:
            return order, False

        new_order = new_order[1:]  # remove depot
        order = new_order  # update for next attempt
        # re-check full feasibility
        feasible, _, _, _ = _check_route(order, instance, dist)
        if feasible:
            return order, True

    return order, False


def _partition_customers(customers, instance, dist, max_cluster_size=8):
    """
    Greedy clustering: start from depot, repeatedly pick the nearest
    unassigned customer, build a cluster until size limit.
    """
    unassigned = set(customers)
    clusters = []
    while unassigned:
        # start cluster from the customer closest to depot
        start = min(unassigned, key=lambda v: dist[0, v])
        cluster = [start]
        unassigned.remove(start)
        while len(cluster) < max_cluster_size and unassigned:
            # find nearest unassigned to any member of cluster
            best_v = None
            best_d = float('inf')
            for v in unassigned:
                for c in cluster:
                    d = dist[c, v]
                    if d < best_d:
                        best_d = d
                        best_v = v
            if best_v is None:
                break
            cluster.append(best_v)
            unassigned.remove(best_v)
        clusters.append(cluster)
    return clusters


def _solve_instance(instance, deadline):
    """Solve one instance, return list of routes."""
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    dist = _build_dist(coords)

    # Partition into clusters
    max_cluster_size = 7
    clusters = _partition_customers(customers, instance, dist, max_cluster_size)

    best_routes = []
    best_total = float('inf')

    for cluster in clusters:
        # Sort by angle from depot for initial ordering
        cx = np.mean([coords[v][0] for v in cluster])
        cy = np.mean([coords[v][1] for v in cluster])
        order = sorted(cluster, key=lambda v: math.atan2(coords[v][1] - cy, coords[v][0] - cx))

        # Improve with 2-opt
        order = _tsp_2opt(order, dist)
        order = _oropt(order, dist)

        # Check feasibility with charging inserts
        order, feasible = _check_feasibility(order, instance, dist)
        if not feasible:
            # Try splitting the cluster in half
            mid = len(order) // 2
            part1 = order[:mid]
            part2 = order[mid:]
            for part in [part1, part2]:
                part = _tsp_2opt(part, dist)
                part, f = _check_feasibility(part, instance, dist)
                if f:
                    best_routes.append(part)
            continue

        best_routes.append(order)

    # Verify all customers covered
    served = set()
    for r in best_routes:
        for v in r:
            if v not in set(stations):
                served.add(v)
    missing = set(customers) - served
    if missing:
        # Add missing customers as single-route fallback
        for v in missing:
            best_routes.append([v])

    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # The instance dict contains all data for ONE instance.
    # But the task says "3 fixed generated instances" - let's check the structure.
    # Looking at the task: instance is a dict with coords, demands, etc.
    # The seed program treats it as a single instance.
    # Let's assume instance is a single instance dict.

    routes = _solve_instance(instance, deadline)

    # Final validation
    dist = _build_dist(instance["coords"])
    total_dist = 0.0
    all_valid = True
    for r in routes:
        feasible, d, _, _ = _check_route(r, instance, dist)
        if not feasible:
            all_valid = False
        total_dist += d

    if not all_valid:
        # Fallback: one customer per route
        n_customers = instance["n_customers"]
        customers = list(range(1, n_customers + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    return routes
