"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from itertools import permutations


def _euclidean(coords):
    n = len(coords)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = math.sqrt(dx * dx + dy * dy)
            d[i, j] = dist
            d[j, i] = dist
    return d


def _dist_matrix(instance):
    coords = instance["coords"]
    return _euclidean(coords)


def _route_time_and_battery(route, instance, dist):
    """
    Simulate a route (list of customer/station node ids, no depot).
    Returns (total_time, total_distance, feasible)
    """
    n = len(route)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])

    # Check capacity
    total_demand = 0
    for node in route:
        if node not in stations:
            total_demand += demands[node]
    if total_demand > capacity:
        return (horizon + 1, 0, False)

    t = 0.0
    battery = battery_max
    total_dist = 0.0
    prev = 0  # depot

    for node in route:
        d = dist[prev, node]
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return (horizon + 1, total_dist + d, False)

        if node in stations:
            # Recharge
            energy_needed = battery_max - battery
            t += energy_needed * recharge_rate
            battery = battery_max
        else:
            # Customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return (horizon + 1, total_dist + d, False)
            t += service[node]

        total_dist += d
        prev = node

    # Return to depot
    d = dist[prev, 0]
    t += d / speed
    battery -= d
    if battery < -1e-9:
        return (horizon + 1, total_dist + d, False)
    if t > horizon + 1e-9:
        return (horizon + 1, total_dist + d, False)

    total_dist += d
    return (t, total_dist, True)


def _route_demand(route, instance):
    stations = set(instance["stations"])
    return sum(instance["demands"][node] for node in route if node not in stations)


def _nearest_neighbor_route(unvisited, instance, dist, start_node=0):
    """Build a route using nearest neighbor heuristic."""
    route = []
    current = start_node
    unvisited_set = set(unvisited)
    
    while unvisited_set:
        # Find nearest unvisited customer
        best = None
        best_d = float('inf')
        for v in unvisited_set:
            d = dist[current, v]
            if d < best_d:
                best_d = d
                best = v
        if best is None:
            break
        route.append(best)
        unvisited_set.remove(best)
        current = best
    
    return route


def _check_route_feasibility(route, instance, dist):
    _, _, feasible = _route_time_and_battery(route, instance, dist)
    return feasible


def _insert_customer(route, customer, instance, dist):
    """
    Try to insert a customer into the route at the best position.
    Returns (new_route, feasible) or (None, False) if no feasible insertion.
    """
    n = len(route)
    best_pos = -1
    best_cost = float('inf')
    
    for pos in range(n + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        # Quick check: demand
        if _route_demand(new_route, instance) > instance["capacity"]:
            continue
        # Check feasibility
        _, total_dist_new, feasible = _route_time_and_battery(new_route, instance, dist)
        if feasible:
            # Compute additional distance
            if pos == 0:
                add_dist = dist[0, customer] + dist[customer, route[0]] - dist[0, route[0]]
            elif pos == n:
                add_dist = dist[route[-1], customer] + dist[customer, 0] - dist[route[-1], 0]
            else:
                add_dist = dist[route[pos-1], customer] + dist[customer, route[pos]] - dist[route[pos-1], route[pos]]
            if add_dist < best_cost:
                best_cost = add_dist
                best_pos = pos
    
    if best_pos == -1:
        return (None, False)
    
    new_route = route[:best_pos] + [customer] + route[best_pos:]
    return (new_route, True)


def _2opt(route, instance, dist):
    """
    Apply 2-opt local search to improve a route.
    Returns improved route.
    """
    n = len(route)
    if n < 2:
        return route
    
    improved = True
    best_route = route[:]
    best_dist = 0.0
    _, best_dist, _ = _route_time_and_battery(best_route, instance, dist)
    
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                _, new_dist, feasible = _route_time_and_battery(new_route, instance, dist)
                if feasible and new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
    
    return best_route


def _solve_instance(instance, deadline):
    """Solve a single instance, returning list of routes."""
    dist = _dist_matrix(instance)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    capacity = instance["capacity"]
    demands = instance["demands"]
    
    # Sort customers by due time for initial heuristic
    customers_sorted = sorted(customers, key=lambda v: instance["due"][v])
    
    # Greedy construction: build routes by inserting customers
    routes = []
    unvisited = set(customers)
    
    while unvisited:
        # Start a new route with the customer with earliest due time
        first = min(unvisited, key=lambda v: instance["due"][v])
        unvisited.remove(first)
        route = [first]
        
        # Try to insert remaining customers
        changed = True
        while changed:
            changed = False
            remaining = list(unvisited)
            # Sort by due time for better insertion
            remaining.sort(key=lambda v: instance["due"][v])
            
            for cust in remaining[:]:
                new_route, feasible = _insert_customer(route, cust, instance, dist)
                if feasible:
                    route = new_route
                    unvisited.remove(cust)
                    changed = True
        
        routes.append(route)
    
    # Apply 2-opt to each route
    for i in range(len(routes)):
        if time.time() > deadline - 5:
            break
        routes[i] = _2opt(routes[i], instance, dist)
    
    return routes


def plan(instance):
    """
    Plan routes for the given instance.
    Returns a list of routes, each a list of node ids.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    routes = _solve_instance(instance, deadline)
    
    # Verify all customers are served exactly once
    all_served = []
    for route in routes:
        for node in route:
            if node > 0 and node <= instance["n_customers"]:
                all_served.append(node)
    
    expected = set(range(1, instance["n_customers"] + 1))
    served_set = set(all_served)
    
    if served_set != expected or len(all_served) != len(expected):
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]
    
    return routes
# EVOLVE-BLOCK-END
