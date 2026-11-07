"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 115.0  # safety margin below 150s limit


def _build_dist_matrix(coords):
    """Build Euclidean distance matrix."""
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _check_route_feasibility(route, dist, instance, n_customers, stations,
                              battery_max, speed, recharge_rate, horizon, capacity):
    """Check if a route is fully feasible. Returns (feasible, distance)."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    total_demand = 0
    for node in route:
        if node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current = 0
    
    for node in route:
        d = dist[current, node]
        time_at += d / speed
        battery -= d
        distance += d
        
        if battery < -1e-9:
            return False, 0.0
        
        if node <= n_customers:
            time_at = max(time_at, ready[node])
            if time_at > due[node] + 1e-9:
                return False, 0.0
            time_at += service[node]
        else:
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = battery_max
        
        current = node
    
    d = dist[current, 0]
    time_at += d / speed
    battery -= d
    distance += d
    
    if battery < -1e-9:
        return False, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0
    
    return True, distance


def _validate_solution(routes, instance, dist):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    served = set()
    total_distance = 0.0
    
    for route in routes:
        feasible, d = _check_route_feasibility(route, dist, instance, n_customers, stations,
                                                battery_max, speed, recharge_rate, horizon, capacity)
        if not feasible:
            return False, 0.0
        total_distance += d
        
        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)
    
    if served != set(range(1, n_customers + 1)):
        return False, 0.0
    
    return True, total_distance


def _compute_total_distance(routes, dist):
    """Compute total distance of routes."""
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist[current, node]
            current = node
        total += dist[current, 0]
    return total


def _greedy_construct(instance, dist):
    """Build initial solution using greedy insertion with time window awareness."""
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    # Sort customers by due time (earliest deadline first)
    customer_order = list(range(1, n_customers + 1))
    customer_order.sort(key=lambda v: due[v])
    
    routes = []
    route_loads = []
    
    def route_battery_feasible(route):
        """Check battery feasibility without time windows."""
        battery = battery_max
        current = 0
        for node in route:
            d = dist[current, node]
            battery -= d
            if battery < -1e-9:
                return False
            if node in stations:
                battery = battery_max
            current = node
        d = dist[current, 0]
        battery -= d
        return battery >= -1e-9
    
    def route_time_feasible(route):
        """Check time feasibility (approximate, ignoring recharge time for speed)."""
        time_at = 0.0
        current = 0
        for node in route:
            d = dist[current, node]
            time_at += d / speed
            if node <= n_customers:
                time_at = max(time_at, ready[node])
                if time_at > due[node] + 1e-9:
                    return False
                time_at += service[node]
            current = node
        d = dist[current, 0]
        time_at += d / speed
        return time_at <= horizon + 1e-9
    
    for cust in customer_order:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')
        
        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                
                if not route_battery_feasible(new_route):
                    continue
                if not route_time_feasible(new_route):
                    continue
                
                # Calculate incremental distance cost
                if pos == 0:
                    prev_node = 0
                    next_node = route[0] if len(route) > 0 else 0
                elif pos == len(route):
                    prev_node = route[-1]
                    next_node = 0
                else:
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                
                old_cost = dist[prev_node, next_node]
                new_cost = dist[prev_node, cust] + dist[cust, next_node]
                cost = new_cost - old_cost
                
                if cost < best_cost:
                    best_cost = cost
                    best_route_idx = r_idx
                    best_insert_pos = pos
        
        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_insert_pos, cust)
            route_loads[best_route_idx] += demands[cust]
    
    return routes


def _optimize_route_order(route, dist, instance, n_customers, stations,
                          battery_max, speed, recharge_rate, horizon, capacity):
    """Optimize the order of nodes within a single route using 2-opt."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    def is_feasible(r):
        return _check_route_feasibility(r, dist, instance, n_customers, stations,
                                        battery_max, speed, recharge_rate, horizon, capacity)[0]
    
    def route_dist(r):
        total = 0.0
        current = 0
        for node in r:
            total += dist[current, node]
            current = node
        total += dist[current, 0]
        return total
    
    best_route = route[:]
    best_dist = route_dist(best_route)
    
    for _ in range(100):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                a = best_route[i - 1] if i > 0 else 0
                b = best_route[i]
                c = best_route[j]
                d = best_route[j + 1] if j + 1 < len(best_route) else 0
                
                old_cost = dist[a, b] + dist[c, d]
                new_cost = dist[a, c] + dist[b, d]
                
                if new_cost < old_cost - 1e-9:
                    new_route = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                    if is_feasible(new_route):
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break
    
    return best_route


def _local_search(routes, instance, dist, max_time=90.0):
    """Improve routes using 2-opt, relocate, and swap moves."""
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])
    
    start_time = time.time()
    current_dist = _compute_total_distance(routes, dist)
    
    def is_feasible(route):
        return _check_route_feasibility(route, dist, instance, n_customers, stations,
                                        battery_max, speed, recharge_rate, horizon, capacity)[0]
    
    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)
    
    def get_total_dist(routes):
        return _compute_total_distance(routes, dist)
    
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        improved = False
        
        # 2-opt within each route
        for r_idx in range(len(routes)):
            route = routes[r_idx]
            if len(route) < 2:
                continue
            for i in range(len(route) - 1):
                if time.time() - start_time > max_time:
                    break
                for j in range(i + 1, len(route)):
                    a = route[i - 1] if i > 0 else 0
                    b = route[i]
                    c = route[j]
                    d = route[j + 1] if j + 1 < len(route) else 0
                    
                    old_cost = dist[a, b] + dist[c, d]
                    new_cost = dist[a, c] + dist[b, d]
                    
                    if new_cost < old_cost - 1e-9:
                        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                        if is_feasible(new_route) and route_load(new_route) <= capacity:
                            routes[r_idx] = new_route
                            current_dist = get_total_dist(routes)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        # Relocate: move a customer from one route to another
        if not improved and time.time() - start_time <= max_time:
            for r1 in range(len(routes)):
                for pos1 in range(len(routes[r1])):
                    if time.time() - start_time > max_time:
                        break
                    cust = routes[r1][pos1]
                    if cust > n_customers:
                        continue
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        if route_load(routes[r2]) + demands[cust] > capacity:
                            continue
                        for pos2 in range(len(routes[r2]) + 1):
                            route1_new = routes[r1][:pos1] + routes[r1][pos1 + 1:]
                            route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                            
                            if is_feasible(route1_new) and is_feasible(route2_new):
                                new_routes = [r[:] for r in routes]
                                new_routes[r1] = route1_new
                                new_routes[r2] = route2_new
                                new_dist = get_total_dist(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # Swap: exchange two customers between routes
        if not improved and time.time() - start_time <= max_time:
            for r1 in range(len(routes)):
                for pos1 in range(len(routes[r1])):
                    if time.time() - start_time > max_time:
                        break
                    cust1 = routes[r1][pos1]
                    if cust1 > n_customers:
                        continue
                    for r2 in range(len(routes)):
                        if r1 == r2:
                            continue
                        for pos2 in range(len(routes[r2])):
                            cust2 = routes[r2][pos2]
                            if cust2 > n_customers:
                                continue
                            
                            # Check capacity
                            load1 = route_load(routes[r1]) - demands[cust1] + demands[cust2]
                            load2 = route_load(routes[r2]) - demands[cust2] + demands[cust1]
                            if load1 > capacity or load2 > capacity:
                                continue
                            
                            route1_new = routes[r1][:]
                            route2_new = routes[r2][:]
                            route1_new[pos1] = cust2
                            route2_new[pos2] = cust1
                            
                            if is_feasible(route1_new) and is_feasible(route2_new):
                                new_routes = [r[:] for r in routes]
                                new_routes[r1] = route1_new
                                new_routes[r2] = route2_new
                                new_dist = get_total_dist(new_routes)
                                if new_dist < current_dist - 1e-9:
                                    routes = new_routes
                                    current_dist = new_dist
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    # Remove empty routes
    routes = [r for r in routes if len(r) > 0]
    
    return routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    
    coords = np.array(instance["coords"], dtype=np.float64)
    dist = _build_dist_matrix(coords)
    
    # Build initial solution
    routes = _greedy_construct(instance, dist)
    
    # Validate
    valid, d = _validate_solution(routes, instance, dist)
    if not valid:
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]
    
    # Optimize route orders
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    
    for r_idx in range(len(routes)):
        if time.time() - start_time > DEADLINE_SECONDS - 15:
            break
        routes[r_idx] = _optimize_route_order(routes[r_idx], dist, instance, n_customers,
                                              stations, battery_max, speed, recharge_rate,
                                              horizon, capacity)
    
    # Local search
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        routes = _local_search(routes, instance, dist, max_time=DEADLINE_SECONDS - 5)
        
        # Re-validate
        valid, d = _validate_solution(routes, instance, dist)
        if not valid:
            customers = list(range(1, instance["n_customers"] + 1))
            routes = [[v] for v in customers]
    
    # Final check
    valid, d = _validate_solution(routes, instance, dist)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]
    
    return routes

# EVOLVE-BLOCK-END
