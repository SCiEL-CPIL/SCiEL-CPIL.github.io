"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random
import heapq

DEADLINE_SECONDS = 140.0  # safety margin below 150s limit


def _build_dist_matrix(coords):
    """Precompute distance matrix using vectorized operations."""
    coords = np.array(coords, dtype=np.float64)
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    return dist_matrix


def _simulate_route(route, dist_matrix, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon,
                    demands, ready, due, service):
    """Simulate a route. Returns (feasible, distance)."""
    total_demand = 0
    for node in route:
        if node <= n_customers:
            total_demand += demands[node]
    if total_demand > 100:  # capacity is always 100
        return False, 0.0

    battery = float(battery_max)
    time_at = 0.0
    distance = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d
        battery -= d
        distance += d

        if battery < -1e-9:
            return False, 0.0

        if node <= n_customers:
            r = ready[node]
            if time_at < r:
                time_at = r
            if time_at > due[node] + 1e-9:
                return False, 0.0
            time_at += service[node]
        elif node in stations:
            energy_needed = battery_max - battery
            time_at += energy_needed * recharge_rate
            battery = float(battery_max)

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d
    battery -= d
    distance += d

    if battery < -1e-9:
        return False, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0

    return True, distance


def _validate_solution(routes, instance):
    """Validate all routes. Returns (valid, total_distance)."""
    n_customers = instance["n_customers"]
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
    stations = set(instance["stations"])
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    served = set()
    total_distance = 0.0

    for route in routes:
        feasible, dist = _simulate_route(route, dist_matrix, n_customers, stations,
                                          battery_max, speed, recharge_rate, horizon,
                                          demands, ready, due, service)
        if not feasible:
            return False, 0.0
        total_distance += dist

        for node in route:
            if node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    """Compute total distance of routes."""
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _greedy_construct(instance):
    """Construct initial solution using greedy insertion."""
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)
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

    # Sort customers by due time
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])

    routes = []
    route_loads = []

    for cust in customers:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]

                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue

                # Calculate insertion cost
                if pos == 0:
                    cost = (dist_matrix[0, cust] + dist_matrix[cust, route[0]] -
                           dist_matrix[0, route[0]])
                elif pos == len(route):
                    cost = (dist_matrix[route[-1], cust] + dist_matrix[cust, 0] -
                           dist_matrix[route[-1], 0])
                else:
                    cost = (dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] -
                           dist_matrix[route[pos-1], route[pos]])

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


def _two_opt_route(route, dist_matrix, n_customers, stations,
                   battery_max, speed, recharge_rate, horizon,
                   demands, ready, due, service, max_iters=50):
    """Optimize route order using 2-opt."""
    if len(route) < 2:
        return route

    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]

                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if not feasible:
                    continue

                new_dist = _compute_total_distance([new_route], dist_matrix)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True
        if not improved:
            break

    return best_route


def _local_search(routes, dist_matrix, instance, deadline):
    """Improve solution using 2-opt and inter-route moves."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        feasible, _ = _simulate_route(route, dist_matrix, n_customers, stations,
                                       battery_max, speed, recharge_rate, horizon,
                                       demands, ready, due, service)
        return feasible

    def route_load(route):
        return sum(demands[node] for node in route if node <= n_customers)

    # First: optimize each route internally
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt_route(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service, max_iters=30
        )

    current_dist = get_total_dist(routes)

    # Inter-route moves
    max_passes = 30
    for _pass in range(max_passes):
        if time.time() > deadline:
            break

        improved = False

        # Try moving customers
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if cust > n_customers:
                    continue

                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue

                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break

                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
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

        if not improved:
            break

    routes = [r for r in routes if len(r) > 0]
    return routes


def _time_window_2opt(route, dist_matrix, n_customers, stations,
                      battery_max, speed, recharge_rate, horizon,
                      demands, ready, due, service):
    """Time-window aware 2-opt: only accept moves that maintain feasibility."""
    if len(route) < 2:
        return route

    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)
    
    for _ in range(20):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                               stations, battery_max, speed,
                                               recharge_rate, horizon,
                                               demands, ready, due, service)
                if feasible:
                    new_dist = _compute_total_distance([new_route], dist_matrix)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break

    return best_route


def _or_opt(route, dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service):
    """OR-opt: move 1-3 consecutive customers to another position."""
    if len(route) < 3:
        return route

    best_route = route[:]
    best_dist = _compute_total_distance([best_route], dist_matrix)
    
    for _ in range(15):
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(len(best_route) - seg_len):
                segment = best_route[i:i+seg_len]
                remaining = best_route[:i] + best_route[i+seg_len:]
                
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    if new_route == best_route:
                        continue
                    
                    feasible, _ = _simulate_route(new_route, dist_matrix, n_customers,
                                                   stations, battery_max, speed,
                                                   recharge_rate, horizon,
                                                   demands, ready, due, service)
                    if feasible:
                        new_dist = _compute_total_distance([new_route], dist_matrix)
                        if new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break

    return best_route


def _swap_customers(routes, dist_matrix, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon,
                    demands, ready, due, service):
    """Try swapping customers between routes."""
    current_dist = _compute_total_distance(routes, dist_matrix)
    best_routes = [r[:] for r in routes]
    
    for r1 in range(len(routes)):
        for r2 in range(r1 + 1, len(routes)):
            for i in range(len(routes[r1])):
                for j in range(len(routes[r2])):
                    c1 = routes[r1][i]
                    c2 = routes[r2][j]
                    
                    if c1 > n_customers or c2 > n_customers:
                        continue
                    
                    # Check capacity constraints
                    load1 = sum(demands[n] for n in routes[r1] if n <= n_customers)
                    load2 = sum(demands[n] for n in routes[r2] if n <= n_customers)
                    
                    if load1 - demands[c1] + demands[c2] > 100:
                        continue
                    if load2 - demands[c2] + demands[c1] > 100:
                        continue
                    
                    new_routes = [r[:] for r in routes]
                    new_routes[r1][i] = c2
                    new_routes[r2][j] = c1
                    
                    feasible1, _ = _simulate_route(new_routes[r1], dist_matrix, n_customers,
                                                    stations, battery_max, speed,
                                                    recharge_rate, horizon,
                                                    demands, ready, due, service)
                    feasible2, _ = _simulate_route(new_routes[r2], dist_matrix, n_customers,
                                                    stations, battery_max, speed,
                                                    recharge_rate, horizon,
                                                    demands, ready, due, service)
                    
                    if feasible1 and feasible2:
                        new_dist = _compute_total_distance(new_routes, dist_matrix)
                        if new_dist < current_dist - 1e-9:
                            best_routes = new_routes
                            current_dist = new_dist
    
    return best_routes


def _relocate_customers(routes, dist_matrix, n_customers, stations,
                        battery_max, speed, recharge_rate, horizon,
                        demands, ready, due, service):
    """Try relocating customers from one route to another."""
    current_dist = _compute_total_distance(routes, dist_matrix)
    best_routes = [r[:] for r in routes]
    
    for r1 in range(len(routes)):
        for i in range(len(routes[r1])):
            cust = routes[r1][i]
            if cust > n_customers:
                continue
            
            for r2 in range(len(routes)):
                if r1 == r2:
                    continue
                
                load2 = sum(demands[n] for n in routes[r2] if n <= n_customers)
                if load2 + demands[cust] > 100:
                    continue
                
                for j in range(len(routes[r2]) + 1):
                    route1_new = routes[r1][:i] + routes[r1][i+1:]
                    route2_new = routes[r2][:j] + [cust] + routes[r2][j:]
                    
                    if len(route1_new) == 0:
                        continue
                    
                    feasible1, _ = _simulate_route(route1_new, dist_matrix, n_customers,
                                                   stations, battery_max, speed,
                                                   recharge_rate, horizon,
                                                   demands, ready, due, service)
                    feasible2, _ = _simulate_route(route2_new, dist_matrix, n_customers,
                                                   stations, battery_max, speed,
                                                   recharge_rate, horizon,
                                                   demands, ready, due, service)
                    
                    if feasible1 and feasible2:
                        new_routes = [r[:] for r in routes]
                        new_routes[r1] = route1_new
                        new_routes[r2] = route2_new
                        new_dist = _compute_total_distance(new_routes, dist_matrix)
                        if new_dist < current_dist - 1e-9:
                            best_routes = new_routes
                            current_dist = new_dist
    
    return [r for r in best_routes if len(r) > 0]


def _advanced_local_search(routes, dist_matrix, instance, deadline):
    """More aggressive local search with multiple operators."""
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    # Apply 2-opt to each route
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _time_window_2opt(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service
        )

    # Apply OR-opt to each route
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _or_opt(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service
        )

    # Inter-route moves
    max_passes = 20
    for _pass in range(max_passes):
        if time.time() > deadline:
            break

        # Try swaps
        new_routes = _swap_customers(routes, dist_matrix, n_customers, stations,
                                     battery_max, speed, recharge_rate, horizon,
                                     demands, ready, due, service)
        if _compute_total_distance(new_routes, dist_matrix) < _compute_total_distance(routes, dist_matrix) - 1e-9:
            routes = new_routes
            continue

        # Try relocations
        new_routes = _relocate_customers(routes, dist_matrix, n_customers, stations,
                                         battery_max, speed, recharge_rate, horizon,
                                         demands, ready, due, service)
        if _compute_total_distance(new_routes, dist_matrix) < _compute_total_distance(routes, dist_matrix) - 1e-9:
            routes = new_routes
            continue

        # No improvement, break
        break

    # Final 2-opt pass
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _time_window_2opt(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service
        )

    return [r for r in routes if len(r) > 0]


def _cluster_based_construct(instance):
    """Construct routes using k-means clustering, then optimize each cluster."""
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations = set(instance["stations"])

    dist_matrix = _build_dist_matrix(coords)

    # Determine number of vehicles needed
    total_demand = sum(demands[1:n_customers+1])
    n_vehicles = max(1, int(np.ceil(total_demand / capacity)))
    n_vehicles = min(n_vehicles, n_customers)

    # Use simple k-means
    customer_coords = coords[1:n_customers+1]
    
    # Initialize centroids randomly
    random.seed(42)
    indices = random.sample(range(n_customers), n_vehicles)
    centroids = customer_coords[indices].copy()

    for _ in range(10):
        # Assign customers to nearest centroid
        labels = np.argmin(np.linalg.norm(customer_coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2), axis=1)
        
        # Update centroids
        new_centroids = []
        for k in range(n_vehicles):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids.append(customer_coords[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        centroids = np.array(new_centroids)

    # Build routes from clusters
    routes = []
    for k in range(n_vehicles):
        mask = labels == k
        cluster_customers = [i + 1 for i in range(n_customers) if mask[i]]
        
        if len(cluster_customers) == 0:
            continue

        # Check capacity
        cluster_demand = sum(demands[c] for c in cluster_customers)
        if cluster_demand > capacity:
            # Split cluster if needed
            half = len(cluster_customers) // 2
            routes.append(cluster_customers[:half])
            routes.append(cluster_customers[half:])
        else:
            routes.append(cluster_customers)

    # Optimize each route
    for r_idx in range(len(routes)):
        routes[r_idx] = _two_opt_route(
            routes[r_idx], dist_matrix, n_customers, stations,
            battery_max, speed, recharge_rate, horizon,
            demands, ready, due, service, max_iters=20
        )

    return routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Build distance matrix once
    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    # Get greedy solution
    routes = _greedy_construct(instance)

    # Validate
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]

    # Try cluster-based construction as alternative
    if time.time() < deadline - 20:
        cluster_routes = _cluster_based_construct(instance)
        valid_cluster, cluster_dist = _validate_solution(cluster_routes, instance)
        if valid_cluster and cluster_dist < dist:
            routes = cluster_routes
            dist = cluster_dist

    # Advanced local search if time permits
    if time.time() < deadline - 15:
        routes = _advanced_local_search(routes, dist_matrix, instance, deadline)

        valid, dist = _validate_solution(routes, instance)
        if not valid:
            customers = list(range(1, instance["n_customers"] + 1))
            routes = [[v] for v in customers]

    # Final check
    valid, dist = _validate_solution(routes, instance)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]

    return routes

# EVOLVE-BLOCK-END
