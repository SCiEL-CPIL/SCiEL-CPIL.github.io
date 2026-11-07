"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 110.0


def _euclidean(p1, p2):
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _build_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _euclidean(coords[i], coords[j])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _simulate_route(route, instance, dist):
    """
    Simulate a route and return (total_distance, is_valid, battery_ok, time_ok).
    route is a list of node ids (customers and/or stations), depot implicit at ends.
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

    n_customers = instance["n_customers"]

    # Check capacity
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return 0.0, False, False, False

    total_dist = 0.0
    battery = battery_max
    current_time = 0.0
    current_node = 0  # depot

    for v in route:
        d = dist[current_node, v]
        total_dist += d
        battery -= d
        if battery < -1e-9:
            return total_dist, False, False, False
        current_time += d / speed

        if v in stations:
            # Recharge: time = (battery_max - battery) * recharge_rate? 
            # Wait, the problem says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This phrasing is ambiguous. Usually it means time to recharge to full.
            # Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # Actually, typically it's (full - current) / rate.
            # Let's look at the variable name: recharge_rate 0.5.
            # If battery is 0, time = 300 * 0.5 = 150? Or 300 / 0.5 = 600?
            # "recharge_rate 0.5" usually implies units of battery per time, or time per unit battery.
            # "adds (battery - energy) * recharge_rate time" -> if energy is 0 (empty), and battery is max?
            # Let's assume the standard interpretation: Time = (battery_max - current_battery) / recharge_rate?
            # Or is recharge_rate the time per unit?
            # Let's look at the text: "recharge_rate 0.5".
            # If it's "time = (max - current) * rate", then with rate 0.5, refilling 300 units takes 150 time.
            # If it's "time = (max - current) / rate", then with rate 0.5, refilling 300 units takes 600 time.
            # Given the horizon is 1000 and distances are up to ~100*sqrt(2) ~ 140 per leg, 25 customers.
            # Total distance for 1 route might be ~500-1000. Time ~500-1000.
            # If recharge takes 150, that's significant. If 600, it's huge.
            # The text says: "adds (battery - energy) * recharge_rate time".
            # "battery" likely refers to the max battery capacity variable name in the instance dict?
            # Instance has "battery": 300.
            # So time = (300 - current_battery) * 0.5.
            # Let's go with this interpretation as it matches the variable names in the prompt.
            recharge_time = (instance["battery"] - battery) * recharge_rate
            current_time += recharge_time
            battery = instance["battery"]
        else:
            # Customer
            if current_time < ready[v]:
                current_time = ready[v]
            if current_time > due[v] + 1e-9:
                return total_dist, False, False, False
            current_time += service[v]

        current_node = v

    # Return to depot
    d = dist[current_node, 0]
    total_dist += d
    battery -= d
    if battery < -1e-9:
        return total_dist, False, False, False
    current_time += d / speed

    if current_time > horizon + 1e-9:
        return total_dist, False, False, False

    return total_dist, True, True, True


def _total_distance(routes, instance, dist):
    total = 0.0
    for route in routes:
        if not route:
            continue
        total += dist[0, route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i], route[i + 1]]
        total += dist[route[-1], 0]
    return total


def _validate_solution(routes, instance, dist):
    """Check if all customers are served exactly once and all routes are valid."""
    n_customers = instance["n_customers"]
    served = set()
    for route in routes:
        for v in route:
            if v <= n_customers:
                if v in served:
                    return False
                served.add(v)
            # Stations can be repeated, but let's check if they are valid nodes
            if v not in instance["stations"] and v > n_customers and v != 0:
                return False
        
    if served != set(range(1, n_customers + 1)):
        return False

    for route in routes:
        _, valid, _, _ = _simulate_route(route, instance, dist)
        if not valid:
            return False
    return True


def _construct_initial_solution(instance, dist):
    """
    Greedy construction: insert customers one by one into the best position.
    Start with empty routes.
    """
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    
    # Sort by due time
    customers.sort(key=lambda v: instance["due"][v])
    
    routes = []
    unassigned = set(customers)
    
    # We will create routes as needed.
    # A simple heuristic: for each customer, try to insert it into an existing route.
    # If no existing route can take it, start a new one.
    # To keep it simple and fast, we might just start with one customer per route,
    # then try to merge/insert.
    
    # Let's try a nearest neighbor approach per route, but with time windows.
    # Actually, let's just do: each customer is its own route initially.
    # Then we try to merge routes if possible.
    
    for c in customers:
        routes.append([c])
        
    # Now try to improve by inserting customers from one route into another.
    # This is complex. Let's stick to a simpler improvement: 2-opt on each route,
    # and then try to merge routes.
    
    # Actually, a better initial solution is to use a savings algorithm or similar.
    # But given the time, let's just use the one-per-route and then do 2-opt.
    # Wait, one per route is valid? 
    # Distance from depot to customer and back.
    # Battery: 2 * dist(depot, c) <= 300? 
    # Max dist in [0,100]^2 is ~141. 2*141 = 282 < 300. So yes, always feasible.
    # Time: 2 * 141 + service (10) = 292 < 1000. Yes.
    
    # So initial solution is always valid.
    return routes


def _two_opt_route(route, instance, dist, max_iterations=100):
    """Apply 2-opt to a single route to minimize distance, maintaining validity."""
    n = len(route)
    if n < 2:
        return route
    
    best_route = route[:]
    best_dist, best_valid, _, _ = _simulate_route(best_route, instance, dist)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                d, valid, _, _ = _simulate_route(new_route, instance, dist)
                if valid and d < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = d
                    improved = True
                    route = new_route[:]
        if not improved:
            break
            
    return best_route


def _try_merge_routes(route1, route2, instance, dist):
    """
    Try to merge two routes into one.
    Return the merged route if valid and shorter, else None.
    """
    # We need to remove the depot legs from the end of route1 and start of route2,
    # and connect them.
    # New route: route1 + route2
    # But we must check if it's valid.
    
    merged = route1 + route2
    d, valid, _, _ = _simulate_route(merged, instance, dist)
    if valid:
        # Check if it's actually shorter than separate
        d1, _, _, _ = _simulate_route(route1, instance, dist)
        d2, _, _, _ = _simulate_route(route2, instance, dist)
        if d < d1 + d2 - 1e-9:
            return merged
    return None


def plan(instance):
    start_time = time.time()
    
    # Precompute distance matrix
    coords = instance["coords"]
    dist = _build_distance_matrix(coords)
    
    # Initial solution
    routes = _construct_initial_solution(instance, dist)
    
    # 2-opt on each route
    for i in range(len(routes)):
        routes[i] = _two_opt_route(routes[i], instance, dist)
    
    # Try to merge routes
    merged = True
    while merged:
        merged = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                merged_route = _try_merge_routes(routes[i], routes[j], instance, dist)
                if merged_route is not None:
                    routes[i] = merged_route
                    routes.pop(j)
                    merged = True
                    break
            if merged:
                break
        
        # After merging, do 2-opt again on the changed routes
        # For simplicity, just do 2-opt on all routes again if anything changed
        if merged:
            for i in range(len(routes)):
                routes[i] = _two_opt_route(routes[i], instance, dist)
        
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
    
    # Final check and validation
    if not _validate_solution(routes, instance, dist):
        # Fallback to one customer per route
        n_customers = instance["n_customers"]
        routes = [[v] for v in range(1, n_customers + 1)]
    
    return routes
# EVOLVE-BLOCK-END
