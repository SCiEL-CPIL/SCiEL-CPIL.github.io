"""Optimized EV routing with time windows and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
import time
import random

DEADLINE_SECONDS = 145.0  # safety margin below 150s limit


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


def _simulate_route(route, dist, instance, n_customers, stations,
                    battery_max, speed, recharge_rate, horizon, capacity):
    """Simulate a route. Returns (feasible, distance, end_time, end_battery)."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    for node in route:
        if node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

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
            return False, 0.0, 0.0, 0.0

        if node <= n_customers:
            time_at = max(time_at, ready[node])
            if time_at > due[node] + 1e-9:
                return False, 0.0, 0.0, 0.0
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
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + 1e-9:
        return False, 0.0, 0.0, 0.0

    return True, distance, time_at, battery


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
        feasible, d, _, _ = _simulate_route(route, dist, instance, n_customers, stations,
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


def _cluster_construct(instance, dist, rng):
    """Cluster-based construction: group customers spatially, then order within clusters."""
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

    customers = list(range(1, n_customers + 1))

    # Estimate max customers per route based on capacity
    avg_demand = np.mean(demands[1:n_customers + 1])
    max_per_route = max(2, int(capacity / avg_demand) + 1)

    # Determine number of routes needed
    n_routes = max(1, int(np.ceil(n_customers / max_per_route)))

    # Spatial clustering using k-means-like approach
    # Initialize cluster centers at depot and spread out
    n_total = len(coords)
    # Use customers for clustering
    cust_coords = coords[1:n_customers + 1]

    # Simple k-means
    k = n_routes
    # Initialize with spread-out points
    indices = list(range(n_customers))
    rng.shuffle(indices)
    centers = [cust_coords[indices[i * n_customers // k]] for i in range(k)]

    for _ in range(20):
        # Assign
        assignments = [[] for _ in range(k)]
        for i, c in enumerate(customers):
            min_d = float('inf')
            best_k = 0
            for ki in range(k):
                d = (cust_coords[i][0] - centers[ki][0]) ** 2 + (cust_coords[i][1] - centers[ki][1]) ** 2
                if d < min_d:
                    min_d = d
                    best_k = ki
            assignments[best_k].append(c)

        # Update centers
        new_centers = []
        for ki in range(k):
            if assignments[ki]:
                cx = np.mean([cust_coords[c - 1][0] for c in assignments[ki]])
                cy = np.mean([cust_coords[c - 1][1] for c in assignments[ki]])
                new_centers.append([cx, cy])
            else:
                new_centers.append(centers[ki])
        centers = new_centers

    # Build routes from clusters, splitting if over capacity
    routes = []
    for ki in range(k):
        cluster = assignments[ki]
        if not cluster:
            continue
        # Sort by angle from cluster center or by due time
        cluster.sort(key=lambda v: due[v])
        # Split if over capacity
        current = []
        load = 0
        for c in cluster:
            if load + demands[c] > capacity:
                if current:
                    routes.append(current)
                current = [c]
                load = demands[c]
            else:
                current.append(c)
                load += demands[c]
        if current:
            routes.append(current)

    # If we have too few routes, that's fine. If some customers are missing, add them
    served = set()
    for r in routes:
        for c in r:
            served.add(c)
    missing = [c for c in customers if c not in served]
    if missing:
        for c in missing:
            routes.append([c])

    # Order each route using nearest neighbor with time window awareness
    final_routes = []
    for route in routes:
        if len(route) <= 1:
            final_routes.append(route)
            continue

        # Nearest neighbor with time window feasibility
        remaining = list(route)
        ordered = []
        current = 0
        time_at = 0.0
        battery = battery_max
        load = 0

        while remaining:
            best_c = None
            best_score = float('inf')
            for c in remaining:
                d = dist[current, c]
                new_time = time_at + d / speed
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                if c <= n_customers:
                    new_time = max(new_time, ready[c])
                    if new_time > due[c] + 1e-9:
                        continue
                    new_time += service[c]
                    new_battery = battery - d
                score = d + 0.1 * max(0, due[c] - new_time)
                if score < best_score:
                    best_score = score
                    best_c = c

            if best_c is None:
                # Pick the one with earliest due
                best_c = min(remaining, key=lambda c: due[c])

            ordered.append(best_c)
            remaining.remove(best_c)
            current = best_c
            time_at += dist[ordered[-2] if len(ordered) > 1 else 0, best_c] / speed if len(ordered) > 1 else dist[0, best_c] / speed
            # Simplified: just track for ordering purposes
            if best_c <= n_customers:
                time_at = max(time_at, ready[best_c]) + service[best_c]

        final_routes.append(ordered)

    return final_routes


def _greedy_construct(instance, dist, rng):
    """Greedy insertion construction as fallback."""
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

    customer_order = list(range(1, n_customers + 1))
    customer_order.sort(key=lambda v: due[v])

    routes = []
    route_loads = []

    def route_feasible(route):
        feasible, _, _, _ = _simulate_route(route, dist, instance, n_customers, stations,
                                            battery_max, speed, recharge_rate, horizon, capacity)
        return feasible

    for cust in customer_order:
        best_route_idx = -1
        best_insert_pos = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue

            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                if not route_feasible(new_route):
                    continue

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


def _simulated_annealing(routes, instance, dist, rng, max_time=100.0):
    """Simulated annealing with OR-opt and 2-opt moves."""
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]
    demands = instance["demands"]

    start_time = time.time()
    current_routes = [r[:] for r in routes]
    current_dist = _compute_total_distance(current_routes, dist)

    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    T = 50.0  # initial temperature
    T_min = 0.01
    cooling = 0.9995
    max_iter = 200000

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist)

    def is_valid(rts):
        valid, d = _validate_solution(rts, instance, dist)
        return valid, d

    def or_opt_move(rts, r1, pos1, length, r2, pos2):
        """Move sequence of 'length' nodes from route r1 at pos1 to route r2 at pos2."""
        if r1 == r2:
            # Intra-route relocate
            if pos1 < pos2:
                pos2 -= length
            seq = rts[r1][pos1:pos1 + length]
            new_r1 = rts[r1][:pos1] + rts[r1][pos1 + length:]
            new_r2 = rts[r2][:pos2] + seq + rts[r2][pos2:]
            return new_r1, new_r2
        else:
            seq = rts[r1][pos1:pos1 + length]
            new_r1 = rts[r1][:pos1] + rts[r1][pos1 + length:]
            new_r2 = rts[r2][:pos2] + seq + rts[r2][pos2:]
            return new_r1, new_r2

    def two_opt_move(rts, r_idx, i, j):
        """2-opt reversal within route r_idx."""
        route = rts[r_idx]
        new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
        return new_route

    for iteration in range(max_iter):
        if time.time() - start_time > max_time:
            break

        T *= cooling
        if T < T_min:
            break

        # Choose move type
        move_type = rng.random()
        n_r = len(current_routes)

        if move_type < 0.4 and n_r >= 2:
            # OR-opt relocate
            r1 = rng.randint(0, n_r - 1)
            if len(current_routes[r1]) < 2:
                continue
            length = rng.choice([1, 2, 3])
            if length > len(current_routes[r1]) - 1:
                length = max(1, len(current_routes[r1]) - 1)
            pos1 = rng.randint(0, len(current_routes[r1]) - length)

            r2 = rng.randint(0, n_r - 1)
            if r2 == r1 and len(current_routes[r1]) <= length:
                continue
            pos2 = rng.randint(0, len(current_routes[r2]) + 1)

            new_r1, new_r2 = or_opt_move(current_routes, r1, pos1, length, r2, pos2)

            # Check capacity
            load1 = sum(demands[n] for n in new_r1 if n <= n_customers)
            load2 = sum(demands[n] for n in new_r2 if n <= n_customers)
            if load1 > capacity or load2 > capacity:
                continue

            new_routes = [r[:] for r in current_routes]
            new_routes[r1] = new_r1
            new_routes[r2] = new_r2

            # Remove empty routes
            new_routes = [r for r in new_routes if len(r) > 0]
            if len(new_routes) != n_r:
                # Re-index if needed - skip for simplicity
                continue

            valid, new_dist = is_valid(new_routes)
            if not valid:
                continue

            delta = new_dist - current_dist
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_routes = [r[:] for r in current_routes]

        elif move_type < 0.7:
            # 2-opt within a route
            r_idx = rng.randint(0, n_r - 1)
            if len(current_routes[r_idx]) < 3:
                continue
            i = rng.randint(0, len(current_routes[r_idx]) - 2)
            j = rng.randint(i + 1, len(current_routes[r_idx]) - 1)

            new_route = two_opt_move(current_routes, r_idx, i, j)
            new_routes = [r[:] for r in current_routes]
            new_routes[r_idx] = new_route

            valid, new_dist = is_valid(new_routes)
            if not valid:
                continue

            delta = new_dist - current_dist
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_routes = [r[:] for r in current_routes]

        else:
            # Swap two customers between routes
            if n_r < 2:
                continue
            r1 = rng.randint(0, n_r - 1)
            r2 = rng.randint(0, n_r - 1)
            if r1 == r2:
                continue
            if len(current_routes[r1]) == 0 or len(current_routes[r2]) == 0:
                continue
            pos1 = rng.randint(0, len(current_routes[r1]) - 1)
            pos2 = rng.randint(0, len(current_routes[r2]) - 1)
            c1 = current_routes[r1][pos1]
            c2 = current_routes[r2][pos2]
            if c1 > n_customers or c2 > n_customers:
                continue

            # Check capacity
            load1 = sum(demands[n] for n in current_routes[r1]) - demands[c1] + demands[c2]
            load2 = sum(demands[n] for n in current_routes[r2]) - demands[c2] + demands[c1]
            if load1 > capacity or load2 > capacity:
                continue

            new_routes = [r[:] for r in current_routes]
            new_routes[r1][pos1] = c2
            new_routes[r2][pos2] = c1

            valid, new_dist = is_valid(new_routes)
            if not valid:
                continue

            delta = new_dist - current_dist
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_dist = current_dist
                    best_routes = [r[:] for r in current_routes]

    return best_routes


def plan(instance):
    """Main entry point."""
    start_time = time.time()
    rng = random.Random(42)

    coords = np.array(instance["coords"], dtype=np.float64)
    dist = _build_dist_matrix(coords)

    # Try multiple construction methods, pick best
    candidates = []

    # Cluster-based construction
    t0 = time.time()
    routes_cluster = _cluster_construct(instance, dist, rng)
    valid, d_cluster = _validate_solution(routes_cluster, instance, dist)
    if valid:
        candidates.append((d_cluster, routes_cluster))

    # Greedy construction
    t0 = time.time()
    routes_greedy = _greedy_construct(instance, dist, rng)
    valid, d_greedy = _validate_solution(routes_greedy, instance, dist)
    if valid:
        candidates.append((d_greedy, routes_greedy))

    if not candidates:
        # Fallback: one customer per route
        customers = list(range(1, instance["n_customers"] + 1))
        routes = [[v] for v in customers]
        return routes

    # Pick best construction
    candidates.sort(key=lambda x: x[0])
    best_dist, best_routes = candidates[0]

    # Simulated annealing
    remaining_time = DEADLINE_SECONDS - (time.time() - start_time) - 5
    if remaining_time > 10:
        best_routes = _simulated_annealing(best_routes, instance, dist, rng, max_time=remaining_time)

    # Final validation
    valid, final_dist = _validate_solution(best_routes, instance, dist)
    if not valid:
        # Try the other candidate
        for d, r in candidates[1:]:
            valid, d2 = _validate_solution(r, instance, dist)
            if valid:
                best_routes = r
                break
        else:
            customers = list(range(1, instance["n_customers"] + 1))
            best_routes = [[v] for v in customers]

    valid, _ = _validate_solution(best_routes, instance, dist)
    if not valid:
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[v] for v in customers]

    return best_routes

# EVOLVE-BLOCK-END
