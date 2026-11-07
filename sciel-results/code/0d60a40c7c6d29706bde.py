"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _dist(coords, i, j):
    return np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)


def _build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
            D[i, j] = d
            D[j, i] = d
    return D


def _simulate_route(route, D, coords, demands, ready, due, service,
                    capacity, battery_cap, speed, recharge_rate, horizon, stations_set):
    """
    Simulate a single route starting and ending at depot (node 0).
    Returns (feasible, total_distance, total_time, final_battery).
    """
    if not route:
        return True, 0.0, 0.0, battery_cap

    # Check demand
    total_demand = sum(demands[v] for v in route)
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    t = 0.0
    battery = battery_cap
    total_dist = 0.0
    prev = 0  # depot

    for v in route:
        d = D[prev, v]
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return False, 0.0, 0.0, 0.0

        if v in stations_set:
            # Charging station: add recharge time, refill battery
            recharge_time = (battery - 0) * recharge_rate  # battery is current level, energy=0? 
            # Wait, the rule says: "a station adds (battery - energy) * recharge_rate time and refills"
            # I need to interpret "energy". Usually, this means the amount of energy *added* or the *current* battery level?
            # Let's re-read: "a station adds (battery - energy) * recharge_rate time and refills"
            # If it refills to full, the energy added is (battery_cap - battery).
            # But the formula is (battery - energy). This is ambiguous.
            # Common interpretation in these problems:
            # 1. "energy" is the battery level before charging. So time = battery * recharge_rate? No, that doesn't make sense physically.
            # 2. "energy" is the energy *consumed*? No.
            # 3. Maybe "energy" is a typo for "0"? i.e. time = battery * recharge_rate?
            # 4. Maybe "energy" is the energy *added*? i.e. time = (battery - (battery - added)) * rate = added * rate?
            #    If it refills to full, added = battery_cap - battery. So time = (battery_cap - battery) * recharge_rate.
            # Let's look at the phrasing again: "adds (battery - energy) * recharge_rate time".
            # If "energy" means the current battery level, then (battery - battery) = 0. That would mean zero time. Unlikely.
            # If "energy" means the energy *to be added*, then (battery - (battery - added)) = added. So time = added * recharge_rate.
            # This makes physical sense: time proportional to energy added.
            # So: energy_added = battery_cap - battery
            # time += energy_added * recharge_rate
            # battery = battery_cap
            
            energy_added = battery_cap - battery
            t += energy_added * recharge_rate
            battery = battery_cap
        else:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return False, 0.0, 0.0, 0.0
            t += service[v]

        total_dist += d
        prev = v

    # Return to depot
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
    deadline = start_time + 145.0  # 5s safety margin for 150s limit

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

    # Pre-check: is any customer unreachable?
    # For now, we assume the instance is feasible.

    # Strategy:
    # 1. Sort customers by due date (EDD).
    # 2. Use a greedy insertion heuristic to build routes.
    #    - Start with an empty list of routes.
    #    - For each customer in EDD order, try to insert it into the best feasible position in any existing route.
    #    - If no existing route can accommodate it, start a new route with just this customer.
    # 3. After initial construction, perform local search (2-opt, or-opt) to improve distance while maintaining feasibility.

    # Sort customers by due date
    customers_sorted = sorted(customers, key=lambda v: due[v])

    routes = []
    route_feasible = [True] * len(routes)  # placeholder, will be updated

    def can_insert(route, customer, pos):
        """
        Check if inserting customer at position pos in route is feasible.
        Returns (feasible, new_route)
        """
        new_route = route[:pos] + [customer] + route[pos:]
        feasible, _, _, _ = _simulate_route(new_route, D, coords, demands, ready, due, service,
                                            capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
        return feasible, new_route

    for cust in customers_sorted:
        best_pos = -1
        best_route_idx = -1
        best_cost = float('inf')

        # Try inserting into existing routes
        for r_idx, route in enumerate(routes):
            # Try all positions in this route
            for pos in range(len(route) + 1):
                feasible, new_route = can_insert(route, cust, pos)
                if feasible:
                    # Calculate cost increase (approximate by new route distance - old route distance)
                    old_dist = _route_distance(route, D)
                    new_dist = _route_distance(new_route, D)
                    cost_increase = new_dist - old_dist
                    if cost_increase < best_cost:
                        best_cost = cost_increase
                        best_pos = pos
                        best_route_idx = r_idx

        if best_route_idx != -1:
            # Insert customer
            route = routes[best_route_idx]
            new_route = route[:best_pos] + [cust] + route[best_pos:]
            routes[best_route_idx] = new_route
        else:
            # Start a new route
            routes.append([cust])

    # Local Search: 2-opt improvement
    # We will iterate over routes and try 2-opt moves within each route.
    # Also try moving a customer from one route to another? That's more complex. Let's stick to intra-route 2-opt first.
    
    improved = True
    while improved and (time.time() < deadline - 5.0):
        improved = False
        for r_idx, route in enumerate(routes):
            if len(route) < 2:
                continue
            route_improved = True
            while route_improved and (time.time() < deadline - 5.0):
                route_improved = False
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        # Reverse segment i..j
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        if new_route == route:
                            continue
                        feasible, _, _, _ = _simulate_route(new_route, D, coords, demands, ready, due, service,
                                                            capacity, battery_cap, speed, recharge_rate, horizon, stations_set)
                        if feasible:
                            old_dist = _route_distance(route, D)
                            new_dist = _route_distance(new_route, D)
                            if new_dist < old_dist - 1e-9:
                                routes[r_idx] = new_route
                                route = new_route
                                route_improved = True
                                improved = True
                                break
                    if route_improved:
                        break

    # Final check: ensure all customers are served exactly once
    all_served = []
    for route in routes:
        all_served.extend(route)
    if sorted(all_served) != sorted(customers):
        # Fallback: return the trivial solution if something went wrong
        return [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
