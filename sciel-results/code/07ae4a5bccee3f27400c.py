"""EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
import heapq
from scipy.optimize import linear_sum_assignment


def _build_dist_matrix(coords):
    """Build Euclidean distance matrix using NumPy."""
    n = len(coords)
    c = np.array(coords, dtype=np.float64)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """
    Simulate a single route. Returns (total_distance, is_valid).
    route: list of node ids (customers and/or stations), depot implicit at start/end.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_dist = 0.0
    t = 0.0
    batt = battery_max
    load = 0
    prev = 0  # depot

    for node in route:
        d = dist[prev][node]
        total_dist += d
        batt -= d
        t += d / speed

        if batt < -1e-9:
            return total_dist, False

        load += demands[node]
        if load > capacity:
            return total_dist, False

        if node in stations_set:
            # Recharge
            if batt < 0:
                batt = 0.0
            energy = battery_max - batt
            t += energy * recharge_rate
            batt = battery_max
        else:
            # Customer
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return total_dist, False
            t += service[node]

        prev = node

    # Return to depot
    d = dist[prev][0]
    total_dist += d
    batt -= d
    t += d / speed

    if batt < -1e-9:
        return total_dist, False
    if t > horizon + 1e-9:
        return total_dist, False

    return total_dist, True


def _validate_solution(routes, dist, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity):
    """Validate that all customers are served exactly once and all routes are feasible."""
    served = set()
    total_dist = 0.0
    for r in routes:
        d, valid = _simulate_route(r, dist, instance, n_customers, stations_set,
                                   battery_max, speed, recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_dist += d
        for node in r:
            if node in stations_set:
                continue
            if node in served:
                return False, 0.0
            served.add(node)

    all_customers = set(range(1, n_customers + 1))
    if served != all_customers:
        return False, 0.0

    return True, total_dist


def _compute_total_distance(routes, dist):
    """Compute total distance of all routes."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        prev = 0
        for node in r:
            total += dist[prev][node]
            prev = node
        total += dist[prev][0]
    return total


def _min_stations_needed(dist, battery_max, battery_remaining):
    """
    For a given distance to travel, determine how many station stops are needed
    and where they should be placed (conceptually). Returns number of stops needed.
    Actually, we need a more nuanced approach. Let's compute the max distance
    we can travel without charging, and how many segments we need.
    """
    max_range = battery_max
    if battery_remaining >= dist:
        return 0
    # We need to figure out if we can make it with some station stops
    # This is complex; let's handle it in the pathfinding
    return -1  # Need more complex logic


def _shortest_path_with_charging(start, end, dist, stations_set, battery_max,
                                  battery_at_start, speed, recharge_rate):
    """
    Find the shortest path from start to end, allowing station visits for charging.
    Returns (path, total_distance, total_time, end_battery) or (None, inf, inf, 0).
    
    We model this as: we can travel from any node to any other node directly.
    At stations, we can recharge. We need battery >= 0 at all times.
    
    State: (node, battery) - but battery is continuous. Let's discretize or use Dijkstra.
    Actually, since we can only gain battery at stations (by refilling to max),
    the battery level only matters as: can we reach the next node?
    
    Let's use Dijkstra where state is (node, battery_level) but we'll track
    the minimum distance to reach (node, battery_level). Since battery is continuous,
    let's instead track (node, min_distance_to_reach_with_at_least_b_b).
    
    Simpler approach: since refilling to max at stations, the key insight is:
    - From any node, if we have enough battery to reach a station, we can go there and refill.
    - So the question is: can we reach end from start, possibly via stations?
    
    Let's do a BFS/Dijkstra on nodes, where the "cost" is distance, and we track
    the battery level. Since we always refill to max at stations, the battery
    when leaving a station is always max. When leaving a non-station, it's whatever
    we had.
    
    State: (node, battery) - but let's limit states. Actually, let's just do:
    For each pair (u, v), we can go directly if battery_at_u >= dist[u][v].
    At stations, battery becomes max after stopping.
    
    Let's use Dijkstra where state is (node, battery) and we track min distance.
    To limit states, we note that battery only takes values that are 
    battery_max - sum of distances since last station.
    
    For simplicity and speed, let's do:
    1. Check direct path: if battery_at_start >= dist[start][end], done.
    2. Otherwise, try via each station: for each station s, check if we can reach s
       from start (battery >= dist[start][s]), then from s (with full battery) to end
       (battery_max >= dist[s][end]).
    3. If not, try via two stations, etc.
    
    For efficiency, let's limit to at most 2 station visits.
    """
    # Direct
    if battery_at_start >= dist[start][end] - 1e-9:
        d = dist[start][end]
        t = d / speed
        batt_end = battery_at_start - d
        return [end], d, t, batt_end
    
    # Via one station
    best = None
    for s in stations_set:
        d1 = dist[start][s]
        if battery_at_start >= d1 - 1e-9:
            t1 = d1 / speed
            # Charge at station
            batt1 = battery_at_start - d1
            if batt1 < 0:
                batt1 = 0
            energy = battery_max - batt1
            t1 += energy * recharge_rate
            batt1 = battery_max
            
            d2 = dist[s][end]
            if batt1 >= d2 - 1e-9:
                t2 = t1 + d2 / speed
                batt2 = batt1 - d2
                total_d = d1 + d2
                if best is None or total_d < best[1]:
                    best = ([s, end], total_d, t2, batt2)
    
    if best is not None:
        return best
    
    # Via two stations
    for s1 in stations_set:
        d1 = dist[start][s1]
        if battery_at_start >= d1 - 1e-9:
            t1 = d1 / speed
            batt1 = battery_at_start - d1
            if batt1 < 0:
                batt1 = 0
            energy = battery_max - batt1
            t1 += energy * recharge_rate
            batt1 = battery_max
            
            for s2 in stations_set:
                if s2 == s1:
                    continue
                d2 = dist[s1][s2]
                if batt1 >= d2 - 1e-9:
                    t2 = t1 + d2 / speed
                    batt2 = batt1 - d2
                    if batt2 < 0:
                        batt2 = 0
                    energy2 = battery_max - batt2
                    t2 += energy2 * recharge_rate
                    batt2 = battery_max
                    
                    d3 = dist[s2][end]
                    if batt2 >= d3 - 1e-9:
                        t3 = t2 + d3 / speed
                        batt3 = batt2 - d3
                        total_d = d1 + d2 + d3
                        if best is None or total_d < best[1]:
                            best = ([s1, s2, end], total_d, t3, batt3)
    
    return best


def _matching_based_cluster(customers, dist, instance, n_customers, stations_set,
                            battery_max, speed, recharge_rate, horizon, capacity):
    """
    Use matching to cluster customers into routes based on proximity.
    Returns list of clusters (each a list of customer ids).
    """
    demands = instance["demands"]
    
    # First, determine how many routes we need based on capacity
    total_demand = sum(demands[c] for c in customers)
    min_routes = max(1, math.ceil(total_demand / capacity))
    
    # We'll create clusters by assigning customers to routes
    # Use a simple approach: sort by angle from depot, then assign to nearest cluster
    
    # Compute angles from depot
    depot_coords = np.array(instance["coords"][0])
    angles = []
    for c in customers:
        cx, cy = instance["coords"][c]
        angle = math.atan2(cy - depot_coords[1], cx - depot_coords[0])
        angles.append((angle, c))
    angles.sort()
    
    # Create initial clusters by dividing by angle
    n_clusters = min(len(customers), max(min_routes, len(customers) // 5))
    clusters = [[] for _ in range(n_clusters)]
    
    for i, (angle, c) in enumerate(angles):
        cluster_idx = i % n_clusters
        # Check capacity
        current_demand = sum(demands[cc] for cc in clusters[cluster_idx])
        if current_demand + demands[c] <= capacity:
            clusters[cluster_idx].append(c)
        else:
            # Find a cluster with space
            placed = False
            for ci in range(n_clusters):
                if ci == cluster_idx:
                    continue
                current_demand = sum(demands[cc] for cc in clusters[ci])
                if current_demand + demands[c] <= capacity:
                    clusters[ci].append(c)
                    placed = True
                    break
            if not placed:
                # Create new cluster
                clusters.append([c])
    
    # Remove empty clusters
    clusters = [c for c in clusters if c]
    
    return clusters


def _nearest_neighbor_route(cluster, dist, instance, n_customers, stations_set,
                            battery_max, speed, recharge_rate, horizon, capacity):
    """
    Build a route for a cluster using nearest neighbor with charging awareness.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    if not cluster:
        return []
    
    route = []
    visited = set()
    current_node = 0
    current_battery = battery_max
    current_time = 0.0
    current_load = 0
    
    remaining = set(cluster)
    
    while remaining:
        best_cost = float('inf')
        best_cust = None
        best_path = None
        
        for cust in remaining:
            # Try to reach cust from current_node
            result = _shortest_path_with_charging(
                current_node, cust, dist, stations_set, battery_max,
                current_battery, speed, recharge_rate
            )
            
            if result is not None:
                path, d, t, batt_end = result
                if current_load + demands[cust] <= capacity:
                    # Check time window
                    t_arrive = current_time + t
                    if t_arrive <= due[cust] + 1e-9:
                        # Cost: distance + penalty for waiting
                        wait = max(0, ready[cust] - t_arrive)
                        cost = d + 0.1 * wait
                        
                        # Also check if we can return to depot after
                        d_back = dist[cust][0]
                        if batt_end >= d_back - 1e-9:
                            if cost < best_cost:
                                best_cost = cost
                                best_cust = cust
                                best_path = path
        
        if best_cust is None:
            # Try without the return check (shouldn't happen often)
            for cust in remaining:
                result = _shortest_path_with_charging(
                    current_node, cust, dist, stations_set, battery_max,
                    current_battery, speed, recharge_rate
                )
                if result is not None:
                    path, d, t, batt_end = result
                    if current_load + demands[cust] <= capacity:
                        t_arrive = current_time + t
                        if t_arrive <= due[cust] + 1e-9:
                            cost = d
                            if cost < best_cost:
                                best_cost = cost
                                best_cust = cust
                                best_path = path
        
        if best_cust is None:
            break
        
        # Add path to route
        for node in best_path:
            route.append(node)
            d = dist[current_node][node]
            current_battery -= d
            current_time += d / speed
            if node in stations_set:
                energy = battery_max - max(0, current_battery)
                current_time += energy * recharge_rate
                current_battery = battery_max
            else:
                current_load += demands[node]
                if current_time < ready[node]:
                    current_time = ready[node]
                current_time += service[node]
            current_node = node
        
        visited.add(best_cust)
        remaining.discard(best_cust)
    
    # Verify route
    d_total, valid = _simulate_route(route, dist, instance, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity)
    if not valid:
        # Fallback: one customer per route
        return [[c] for c in cluster]
    
    return route


def _construct_routes_matching(customers, dist, instance, n_customers, stations_set,
                               battery_max, speed, recharge_rate, horizon, capacity):
    """
    Construct routes using matching-based clustering.
    """
    clusters = _matching_based_cluster(customers, dist, instance, n_customers,
                                       stations_set, battery_max, speed, recharge_rate,
                                       horizon, capacity)
    
    routes = []
    for cluster in clusters:
        result = _nearest_neighbor_route(cluster, dist, instance, n_customers,
                                         stations_set, battery_max, speed, recharge_rate,
                                         horizon, capacity)
        if isinstance(result, list) and (not result or isinstance(result[0], int)):
            routes.append(result)
        elif isinstance(result, list):
            # Fallback returned multiple routes
            routes.extend(result)
    
    return routes


def _savings_construct(customers, dist, instance, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity):
    """
    Savings-based construction: start with individual routes, merge if beneficial.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    
    # Start with each customer as a separate route
    routes = [[c] for c in customers]
    
    # Compute savings for merging pairs
    # Savings = dist[0][i] + dist[0][j] - dist[i][j]
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    savings.sort(reverse=True)
    
    # Route index for each customer
    cust_route = {c: i for i, c in enumerate(customers)}
    
    for s, ci, cj in savings:
        if s <= 0:
            break
        
        ri = cust_route.get(ci)
        rj = cust_route.get(cj)
        
        if ri is None or rj is None or ri == rj:
            continue
        
        # Try to merge route rj into route ri
        # ci should be at the end of ri, cj at the start of rj (or vice versa)
        route_i = routes[ri]
        route_j = routes[rj]
        
        if not route_i or not route_j:
            continue
        
        # Check if ci is at an end of route_i
        if ci == route_i[0]:
            # ci is at start, so we can prepend route_j
            # New route: route_j + route_i (but ci must be adjacent to cj)
            # Actually, we need cj to be adjacent to ci
            # If cj is at end of route_j, then new route = route_j + route_i
            if cj == route_j[-1]:
                new_route = route_j + route_i
            elif cj == route_j[0]:
                new_route = list(reversed(route_j)) + route_i
            else:
                continue
        elif ci == route_i[-1]:
            if cj == route_j[0]:
                new_route = route_i + route_j
            elif cj == route_j[-1]:
                new_route = route_i + list(reversed(route_j))
            else:
                continue
        else:
            continue
        
        # Check capacity
        total_demand = sum(demands[c] for c in new_route if c not in stations_set)
        if total_demand > capacity:
            continue
        
        # Check feasibility
        d_new, valid = _simulate_route(new_route, dist, instance, n_customers,
                                       stations_set, battery_max, speed, recharge_rate,
                                       horizon, capacity)
        if valid:
            routes[ri] = new_route
            routes[rj] = []
            for c in new_route:
                if c not in stations_set:
                    cust_route[c] = ri
    
    # Remove empty routes
    routes = [r for r in routes if r]
    
    return routes


def _two_opt(route, dist, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity):
    """Apply 2-opt improvement to a single route."""
    n = len(route)
    if n < 3:
        return route
    
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d_new, valid = _simulate_route(new_route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
                if valid:
                    d_old, _ = _simulate_route(route, dist, instance, n_customers,
                                               stations_set, battery_max, speed,
                                               recharge_rate, horizon, capacity)
                    if d_new < d_old - 1e-9:
                        route = new_route
                        improved = True
    return route


def _or_opt(route, dist, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity):
    """Apply OR-opt (move 1-3 consecutive nodes) improvement."""
    n = len(route)
    if n < 4:
        return route
    
    improved = True
    while improved:
        improved = False
        for length in [1, 2, 3]:
            for i in range(n - length + 1):
                segment = route[i:i + length]
                remaining = route[:i] + route[i + length:]
                
                for j in range(len(remaining) + 1):
                    new_route = remaining[:j] + segment + remaining[j:]
                    if new_route == route:
                        continue
                    d_new, valid = _simulate_route(new_route, dist, instance, n_customers,
                                                   stations_set, battery_max, speed,
                                                   recharge_rate, horizon, capacity)
                    if valid:
                        d_old, _ = _simulate_route(route, dist, instance, n_customers,
                                                   stations_set, battery_max, speed,
                                                   recharge_rate, horizon, capacity)
                        if d_new < d_old - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    return route


def _local_search(routes, dist, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity,
                  time_limit=30.0):
    """Apply 2-opt, OR-opt, and relocate moves."""
    start = time.time()
    improved = True
    
    while improved and (time.time() - start) < time_limit:
        improved = False
        
        # 2-opt within each route
        for ri in range(len(routes)):
            if (time.time() - start) > time_limit:
                return routes
            routes[ri] = _two_opt(
                routes[ri], dist, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity
            )
        
        # OR-opt within each route
        for ri in range(len(routes)):
            if (time.time() - start) > time_limit:
                return routes
            routes[ri] = _or_opt(
                routes[ri], dist, instance, n_customers, stations_set,
                battery_max, speed, recharge_rate, horizon, capacity
            )
        
        # Relocate moves
        for ri in range(len(routes)):
            for pos in range(len(routes[ri])):
                if (time.time() - start) > time_limit:
                    return routes
                cust = routes[ri][pos]
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos2 in range(len(routes[rj]) + 1):
                        new_ri = routes[ri][:pos] + routes[ri][pos + 1:]
                        new_rj = routes[rj][:pos2] + [cust] + routes[rj][pos2:]
                        
                        valid_ri = True
                        valid_rj = True
                        d_ri = 0.0
                        d_rj = 0.0
                        
                        if new_ri:
                            d_ri, valid_ri = _simulate_route(new_ri, dist, instance,
                                                             n_customers, stations_set,
                                                             battery_max, speed,
                                                             recharge_rate, horizon, capacity)
                        if new_rj:
                            d_rj, valid_rj = _simulate_route(new_rj, dist, instance,
                                                             n_customers, stations_set,
                                                             battery_max, speed,
                                                             recharge_rate, horizon, capacity)
                        
                        if valid_ri and valid_rj:
                            old_total = _compute_total_distance([routes[ri], routes[rj]], dist)
                            new_total = d_ri + d_rj
                            if new_total < old_total - 1e-9:
                                routes[ri] = new_ri
                                routes[rj] = new_rj
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    
    return routes


def _simulated_annealing(routes, dist, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity,
                         time_limit=20.0):
    """Apply simulated annealing to improve routes."""
    start = time.time()
    current_routes = [list(r) for r in routes]
    current_dist = _compute_total_distance(current_routes, dist)
    best_routes = [list(r) for r in current_routes]
    best_dist = current_dist
    
    T = 50.0
    cooling = 0.995
    
    while (time.time() - start) < time_limit and T > 0.1:
        new_routes = [list(r) for r in current_routes]
        move_type = random.random()
        
        if move_type < 0.5:
            ri = random.randint(0, len(new_routes) - 1)
            route = new_routes[ri]
            if len(route) >= 3:
                i, j = sorted(random.sample(range(len(route)), 2))
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                d, valid = _simulate_route(new_route, dist, instance, n_customers,
                                           stations_set, battery_max, speed,
                                           recharge_rate, horizon, capacity)
                if valid:
                    new_routes[ri] = new_route
        else:
            ri = random.randint(0, len(new_routes) - 1)
            if len(new_routes[ri]) > 0:
                pos = random.randint(0, len(new_routes[ri]) - 1)
                cust = new_routes[ri][pos]
                rj = random.randint(0, len(new_routes) - 1)
                pos2 = random.randint(0, len(new_routes[rj]))
                new_ri = new_routes[ri][:pos] + new_routes[ri][pos + 1:]
                new_rj = new_routes[rj][:pos2] + [cust] + new_routes[rj][pos2:]
                
                valid_ri = True
                valid_rj = True
                d_ri = 0.0
                d_rj = 0.0
                if new_ri:
                    d_ri, valid_ri = _simulate_route(new_ri, dist, instance, n_customers,
                                                     stations_set, battery_max, speed,
                                                     recharge_rate, horizon, capacity)
                if new_rj:
                    d_rj, valid_rj = _simulate_route(new_rj, dist, instance, n_customers,
                                                     stations_set, battery_max, speed,
                                                     recharge_rate, horizon, capacity)
                
                if valid_ri and valid_rj:
                    new_routes[ri] = new_ri
                    new_routes[rj] = new_rj
        
        new_dist = _compute_total_distance(new_routes, dist)
        delta = new_dist - current_dist
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_routes = new_routes
            current_dist = new_dist
            if current_dist < best_dist:
                best_routes = [list(r) for r in current_routes]
                best_dist = current_dist
        
        T *= cooling
    
    return best_routes


def plan(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    customers = list(range(1, n_customers + 1))
    
    # Build distance matrix
    dist = _build_dist_matrix(coords)
    
    # Deadline guard
    deadline = time.time() + 140.0
    
    # Phase 1: Try multiple construction methods and pick the best
    best_routes = None
    best_dist = float('inf')
    
    # Method 1: Matching-based clustering
    routes1 = _construct_routes_matching(customers, dist, instance, n_customers, stations,
                                         battery_cap, speed, recharge_rate, horizon, capacity)
    valid1, dist1 = _validate_solution(routes1, dist, instance, n_customers, stations,
                                       battery_cap, speed, recharge_rate, horizon, capacity)
    if valid1 and dist1 < best_dist:
        best_dist = dist1
        best_routes = [list(r) for r in routes1]
    
    # Method 2: Savings-based
    if time.time() < deadline - 20.0:
        routes2 = _savings_construct(customers, dist, instance, n_customers, stations,
                                     battery_cap, speed, recharge_rate, horizon, capacity)
        valid2, dist2 = _validate_solution(routes2, dist, instance, n_customers, stations,
                                           battery_cap, speed, recharge_rate, horizon, capacity)
        if valid2 and dist2 < best_dist:
            best_dist = dist2
            best_routes = [list(r) for r in routes2]
    
    # Fallback if no valid solution
    if best_routes is None:
        best_routes = [[v] for v in customers]
        valid, best_dist = _validate_solution(best_routes, dist, instance, n_customers,
                                              stations, battery_cap, speed, recharge_rate,
                                              horizon, capacity)
        if not valid:
            # Shouldn't happen, but just in case
            best_routes = [[v] for v in customers]
    
    # Phase 2: Local search
    if time.time() < deadline - 10.0:
        best_routes = _local_search(
            best_routes, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            time_limit=max(1.0, deadline - time.time() - 10.0)
        )
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 5.0:
        best_routes = _simulated_annealing(
            best_routes, dist, instance, n_customers, stations,
            battery_cap, speed, recharge_rate, horizon, capacity,
            time_limit=max(1.0, deadline - time.time() - 5.0)
        )
    
    # Final validation
    valid, total_dist = _validate_solution(
        best_routes, dist, instance, n_customers, stations,
        battery_cap, speed, recharge_rate, horizon, capacity
    )
    if not valid:
        best_routes = [[v] for v in customers]
    
    return best_routes
# EVOLVE-BLOCK-END
