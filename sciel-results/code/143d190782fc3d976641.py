"""Rewritten program: EVRP with time windows, battery, and charging stations."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import time
import math
import random


def _euclidean(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(coords[i], coords[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate_route(route, instance, D):
    """Simulate a route (list of node ids, depot implicit at both ends).
    Returns (total_distance, feasible, arrival_time, final_battery)."""
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
    n_customers = instance["n_customers"]

    total_dist = 0.0
    battery = battery_cap
    t = 0.0
    total_demand = 0

    prev = 0  # depot
    for node in route:
        d = D[prev, node]
        total_dist += d
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return total_dist, False, t, battery

        if node in stations:
            # Recharge: adds (battery) * recharge_rate time? 
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # energy is 0 (station doesn't consume demand). So adds battery * recharge_rate?
            # Wait, "refills" means battery becomes full.
            # The time added is (battery - energy) * recharge_rate? No, usually charging time = (full - current) / rate.
            # Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # If energy is demand consumed, at station energy=0. So time += battery * recharge_rate?
            # That seems odd. Usually charge time = (capacity - current) / rate.
            # Let's assume standard: time += (battery_cap - battery) / recharge_rate.
            # But the text says "(battery - energy) * recharge_rate". If recharge_rate is 0.5, and battery is 100, time += 50?
            # Let's check units. battery is energy. recharge_rate is 0.5.
            # If rate is energy/time, then time = energy/rate.
            # If the formula is literally (battery - energy) * recharge_rate, and recharge_rate is 0.5,
            # then time = 0.5 * battery.
            # Let's stick to the literal text: time += (battery - 0) * recharge_rate.
            # Then battery = battery_cap.
            t += battery * recharge_rate
            battery = battery_cap
        else:
            # Customer
            total_demand += demands[node]
            if total_demand > capacity:
                return total_dist, False, t, battery
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False, t, battery
            t += service[node]

        prev = node

    # Return to depot
    d = D[prev, 0]
    total_dist += d
    battery -= d
    t += d / speed
    if battery < -1e-9:
        return total_dist, False, t, battery
    if t > horizon + 1e-9:
        return total_dist, False, t, battery

    return total_dist, True, t, battery


def _solve_tsp_dp(nodes, D, start, end, ready, due, service, battery_cap, recharge_rate, speed, horizon, capacity, demands, stations):
    """Solve TSP with time windows and battery using DP.
    nodes: list of customer/station ids to visit.
    Returns (route, cost) or None."""
    if not nodes:
        return [], 0.0

    n = len(nodes)
    if n > 20:
        # Fallback to greedy or simpler heuristic for large n
        # For now, return None to signal failure
        return None, float('inf')

    # State: (mask, last_node_index)
    # mask: bitmask of visited nodes
    # last_node_index: index in nodes list of the last visited node
    # Value: (cost, time, battery) - we want min cost, but must satisfy constraints.
    # Since constraints are complex, we store the best (time, battery) for each (mask, last) that minimizes cost?
    # Actually, for a given (mask, last), a state with lower time and higher battery is always better.
    # We can use Pareto fronts, but for simplicity, let's try minimizing cost first, and check feasibility.
    # Better: DP over (mask, last) storing the best (time, battery) that minimizes cost?
    # No, cost is additive. Time and battery are state variables.
    # Let's use a dictionary: dp[mask][last] = list of (cost, time, battery) Pareto-optimal states.

    # To keep it simple and fast for n<=15, we can just store the best cost for each (mask, last)
    # and check feasibility at the end? No, feasibility depends on path.
    
    # Let's use a simpler approach: Held-Karp with state (mask, last) -> (cost, time, battery)
    # We assume that for a given (mask, last), the state with minimum cost is not necessarily the one that leads to a feasible solution.
    # However, for small n, we can try all paths? No, that's n!.
    
    # Let's try a different approach: since n is small (<=10 per route in practice), we can use a simple recursive search with pruning.
    
    # Actually, let's stick to a simpler heuristic for the whole thing:
    # 1. Cluster customers.
    # 2. For each cluster, find a feasible route using a simple insertion or greedy method.
    
    pass


def _greedy_route(cluster, instance, D):
    """Build a feasible route for a cluster of customers using greedy nearest neighbor with time/battery checks."""
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
    
    route = []
    remaining = set(cluster)
    prev = 0
    t = 0.0
    battery = battery_cap
    total_demand = 0
    total_dist = 0.0

    while remaining:
        best_node = None
        best_dist = float('inf')
        for node in remaining:
            d = D[prev, node]
            new_battery = battery - d
            if new_battery < 0:
                continue
            new_t = t + d / speed
            if node in stations:
                new_t += new_battery * recharge_rate
                new_battery = battery_cap
            else:
                if new_t < ready[node]:
                    new_t = ready[node]
                if new_t > due[node]:
                    continue
                if total_demand + demands[node] > capacity:
                    continue
                new_t += service[node]
            
            # Heuristic: prefer nearest feasible node
            if d < best_dist:
                best_dist = d
                best_node = node

        if best_node is None:
            # Try to insert a station to recharge
            for st in stations:
                d = D[prev, st]
                new_battery = battery - d
                if new_battery < 0:
                    continue
                new_t = t + d / speed
                new_t += new_battery * recharge_rate
                new_battery = battery_cap
                # Check if we can reach any remaining customer from station
                for node in remaining:
                    d2 = D[st, node]
                    if new_battery - d2 < 0:
                        continue
                    t2 = new_t + d2 / speed
                    if t2 < ready[node]:
                        t2 = ready[node]
                    if t2 > due[node]:
                        continue
                    if total_demand + demands[node] > capacity:
                        continue
                    # Feasible: go to station, then to node
                    # For now, just add station to route and continue
                    route.append(st)
                    prev = st
                    t = new_t
                    battery = new_battery
                    total_dist += d
                    break
                else:
                    continue
                break
            else:
                # No feasible station, break (infeasible)
                break

        # Add best_node to route
        d = D[prev, best_node]
        route.append(best_node)
        total_dist += d
        battery -= d
        t += d / speed
        
        if best_node in stations:
            t += battery * recharge_rate
            battery = battery_cap
        else:
            if t < ready[best_node]:
                t = ready[best_node]
            t += service[best_node]
            total_demand += demands[best_node]
            remaining.remove(best_node)
        
        prev = best_node

    # Return to depot
    d = D[prev, 0]
    total_dist += d
    battery -= d
    t += d / speed

    # Check feasibility
    if battery < 0 or t > horizon:
        return None, float('inf')

    return route, total_dist


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    D = _dist_matrix(coords)

    # 1. Cluster customers using K-Means or simple distance-based clustering
    customers = list(range(1, n_customers + 1))
    
    # Simple clustering: sort by angle from depot, then split into groups
    angles = [math.atan2(coords[c][1] - coords[0][1], coords[c][0] - coords[0][0]) for c in customers]
    customers_sorted = [c for c, _ in sorted(zip(customers, angles))]
    
    # Split into ~5 clusters
    n_clusters = 5
    cluster_size = len(customers_sorted) // n_clusters
    clusters = []
    for i in range(n_clusters):
        start = i * cluster_size
        end = start + cluster_size if i < n_clusters - 1 else len(customers_sorted)
        clusters.append(customers_sorted[start:end])

    # 2. For each cluster, build a route using greedy method
    routes = []
    for cluster in clusters:
        route, cost = _greedy_route(cluster, instance, D)
        if route is not None:
            routes.append(route)
        else:
            # Fallback: serve each customer individually
            for c in cluster:
                routes.append([c])

    # 3. Verify all routes are feasible
    final_routes = []
    for route in routes:
        dist, feasible, _, _ = _simulate_route(route, instance, D)
        if feasible:
            final_routes.append(route)
        else:
            # Fallback: split into single customers
            for c in route:
                final_routes.append([c])

    # 4. Ensure all customers are served exactly once
    served = set()
    for route in final_routes:
        for c in route:
            if c in range(1, n_customers + 1):
                served.add(c)
    
    missing = set(range(1, n_customers + 1)) - served
    for c in missing:
        final_routes.append([c])

    # 5. Time check
    if time.time() > deadline:
        return final_routes

    # 6. Local improvement: 2-opt within routes
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i, route in enumerate(final_routes):
            if len(route) < 2:
                continue
            for a in range(len(route) - 1):
                for b in range(a + 1, len(route)):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    dist, feasible, _, _ = _simulate_route(new_route, instance, D)
                    old_dist, _, _, _ = _simulate_route(route, instance, D)
                    if feasible and dist < old_dist - 1e-9:
                        final_routes[i] = new_route
                        improved = True

    return final_routes
# EVOLVE-BLOCK-END
