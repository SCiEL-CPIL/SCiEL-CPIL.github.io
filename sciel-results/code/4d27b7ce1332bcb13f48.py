# EVOLVE-BLOCK-START
import numpy as np
import math
import time


def plan(instance):
    t_start = time.time()
    deadline = t_start + 130.0  # safety margin before 150s kill

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

    n_nodes = len(coords)

    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    customers = list(range(1, n_customers + 1))

    # Check if we need charging at all
    max_dist_from_depot = max(dist[0][c] for c in customers)
    if 2 * max_dist_from_depot + sum(demands[c] for c in customers) <= battery_cap:
        # No charging needed, simple clustering
        pass
    else:
        # Charging may be needed
        pass

    # Cluster customers into groups using simple spatial clustering
    # Target number of routes based on capacity
    total_demand = sum(demands[c] for c in customers)
    n_routes_target = max(1, int(math.ceil(total_demand / capacity)))
    # Also consider distance: try to keep routes short
    n_routes_target = max(n_routes_target, 4)
    n_routes_target = min(n_routes_target, 8)

    # Simple k-means-like clustering
    def cluster_customers(custs, k):
        if k >= len(custs):
            return [[c] for c in custs]
        # Initialize centers using farthest point sampling
        centers = [custs[0]]
        for _ in range(k - 1):
            best_c = None
            best_min_d = -1
            for c in custs:
                min_d = min(dist[c][coords_idx_to_node(coords, centers[-1])] if False else dist[c][centers[-1]] for _ in [0])
                # Actually compute min distance to all existing centers
                min_d = min(dist[c][cc] for cc in centers)
                if min_d > best_min_d:
                    best_min_d = min_d
                    best_c = c
            centers.append(best_c)

        # Assign customers to nearest center
        for _ in range(20):  # iterations
            assignment = [[] for _ in range(k)]
            for c in custs:
                best_k = 0
                best_d = float('inf')
                for ki, cc in enumerate(centers):
                    d = dist[c][cc]
                    if d < best_d:
                        best_d = d
                        best_k = ki
                assignment[best_k].append(c)
            # Update centers
            for ki in range(k):
                if assignment[ki]:
                    # Use mean position
                    xs = np.mean([coords[c][0] for c in assignment[ki]])
                    ys = np.mean([coords[c][1] for c in assignment[ki]])
                    # Find closest customer to mean
                    best_c = None
                    best_d = float('inf')
                    for c in assignment[ki]:
                        d = math.hypot(coords[c][0] - xs, coords[c][1] - ys)
                        if d < best_d:
                            best_d = d
                            best_c = c
                    centers[ki] = best_c
        return assignment

    # Try different numbers of routes and pick best
    best_routes = None
    best_score = float('inf')

    for n_r in range(max(1, n_routes_target - 2), n_routes_target + 3):
        if n_r > len(customers):
            n_r = len(customers)
        if time.time() > deadline - 30:
            break

        clusters = cluster_customers(customers, n_r)

        # Filter empty clusters
        clusters = [cl for cl in clusters if cl]

        # For each cluster, build a route
        routes = []
        for cl in clusters:
            route = optimize_route(cl, dist, demands, ready, due, service,
                                   stations, battery_cap, speed, recharge_rate, horizon)
            if route is not None:
                routes.append(route)
            else:
                # Fallback: one customer per route
                for c in cl:
                    routes.append([c])

        # Score
        score = 0.0
        valid = True
        for route in routes:
            r_score, r_valid = evaluate_route(route, dist, demands, ready, due, service,
                                              stations, battery_cap, speed, recharge_rate, horizon)
            if not r_valid:
                valid = False
                break
            score += r_score

        if valid and score < best_score:
            best_score = score
            best_routes = [list(r) for r in routes]

        if time.time() > deadline - 20:
            break

    # If no valid solution found, fall back to one customer per route
    if best_routes is None:
        best_routes = [[c] for c in customers]

    return best_routes


def coords_idx_to_node(coords, idx):
    """Helper - not actually used properly, placeholder."""
    return idx


def optimize_route(customers_in_route, dist, demands, ready, due, service,
                   stations, battery_cap, speed, recharge_rate, horizon):
    """Build and optimize a route for a set of customers."""
    if not customers_in_route:
        return []

    # Start with nearest-neighbor heuristic
    route = nearest_neighbor(customers_in_route, dist, demands, ready, due, service,
                             stations, battery_cap, speed, recharge_rate, horizon)
    if route is None:
        return None

    # 2-opt improvement
    improved = two_opt(route, dist, demands, ready, due, service,
                       stations, battery_cap, speed, recharge_rate, horizon)
    if improved is not None:
        route = improved

    # Try or-opt (move single customer to better position)
    improved = or_opt(route, dist, demands, ready, due, service,
                      stations, battery_cap, speed, recharge_rate, horizon)
    if improved is not None:
        route = improved

    # Final check
    _, valid = evaluate_route(route, dist, demands, ready, due, service,
                              stations, battery_cap, speed, recharge_rate, horizon)
    if not valid:
        return None

    return route


def nearest_neighbor(customers_in_route, dist, demands, ready, due, service,
                     stations, battery_cap, speed, recharge_rate, horizon):
    """Build route using nearest neighbor with time window feasibility."""
    remaining = set(customers_in_route)
    route = []
    current = 0  # depot
    current_time = 0.0
    current_battery = battery_cap
    current_load = 0

    while remaining:
        best_c = None
        best_dist = float('inf')
        for c in remaining:
            d = dist[current][c]
            # Check if we can reach c with current battery (might need charging)
            if d <= current_battery:
                if d < best_dist:
                    best_dist = d
                    best_c = c
            else:
                # Need to charge first - skip for now, try later
                pass

        if best_c is None:
            # All remaining need charging; pick the one with earliest due
            best_c = min(remaining, key=lambda c: due[c])
            best_dist = dist[current][best_c]

        # Check time window
        arrive_time = current_time + best_dist / speed
        if arrive_time > due[best_c]:
            # Try next best
            sorted_custs = sorted(remaining, key=lambda c: (due[c], dist[current][c]))
            found = False
            for c in sorted_custs:
                d = dist[current][c]
                arrive_t = current_time + d / speed
                if arrive_t <= due[c] and d <= current_battery:
                    best_c = c
                    best_dist = d
                    found = True
                    break
            if not found:
                # Pick the one with most slack
                best_c = min(remaining, key=lambda c: due[c] - (current_time + dist[current][c] / speed))
                best_dist = dist[current][best_c]

        route.append(best_c)
        remaining.remove(best_c)

        # Update state
        current_time += best_dist / speed
        current_battery -= best_dist
        current_time = max(current_time, ready[best_c])
        current_time += service[best_c]
        current_load += demands[best_c]
        current = best_c

    # Check if we can return to depot
    d_back = dist[current][0]
    if d_back > current_battery:
        # Need charging - insert station
        # For now, mark as potentially invalid
        pass

    return route


def two_opt(route, dist, demands, ready, due, service,
            stations, battery_cap, speed, recharge_rate, horizon):
    """Improve route using 2-opt."""
    if len(route) < 2:
        return route

    best_route = list(route)
    best_score, best_valid = evaluate_route(best_route, dist, demands, ready, due, service,
                                            stations, battery_cap, speed, recharge_rate, horizon)
    if not best_valid:
        return None

    improved = True
    max_iters = 50
    for _ in range(max_iters):
        if not improved:
            break
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                # Reverse segment i..j
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                s, v = evaluate_route(new_route, dist, demands, ready, due, service,
                                      stations, battery_cap, speed, recharge_rate, horizon)
                if v and s < best_score - 1e-9:
                    best_score = s
                    best_route = new_route
                    improved = True
    return best_route


def or_opt(route, dist, demands, ready, due, service,
           stations, battery_cap, speed, recharge_rate, horizon):
    """Try moving single customers to better positions."""
    if len(route) < 2:
        return route

    best_route = list(route)
    best_score, best_valid = evaluate_route(best_route, dist, demands, ready, due, service,
                                            stations, battery_cap, speed, recharge_rate, horizon)
    if not best_valid:
        return None

    improved = True
    max_iters = 20
    for _ in range(max_iters):
        if not improved:
            break
        improved = False
        for i in range(len(best_route)):
            c = best_route[i]
            # Remove c
            base = best_route[:i] + best_route[i+1:]
            # Try inserting at each position
            for j in range(len(base) + 1):
                if j == i or (j == i + 1 and i < len(best_route)):
                    continue
                new_route = base[:j] + [c] + base[j:]
                s, v = evaluate_route(new_route, dist, demands, ready, due, service,
                                      stations, battery_cap, speed, recharge_rate, horizon)
                if v and s < best_score - 1e-9:
                    best_score = s
                    best_route = new_route
                    improved = True
                    break
    return best_route


def evaluate_route(route, dist, demands, ready, due, service,
                   stations, battery_cap, speed, recharge_rate, horizon):
    """
    Evaluate a route (list of customer nodes, depot implicit at both ends).
    Returns (total_distance, is_valid).
    Automatically inserts charging stations where needed.
    """
    if not route:
        return 0.0, True

    total_dist = 0.0
    current_node = 0
    current_time = 0.0
    current_battery = battery_cap

    # Build the actual sequence with potential charging stops
    actual_seq = []
    for c in route:
        d = dist[current_node][c]
        if d > current_battery:
            # Need to charge: find best station
            station_seq = find_charging_stops(current_node, c, current_battery, dist,
                                              stations, battery_cap)
            if station_seq is None:
                return float('inf'), False
            for s in station_seq:
                ds = dist[current_node][s]
                total_dist += ds
                current_time += ds / speed
                current_battery -= ds
                # At station: recharge
                charge_time = (battery_cap - current_battery) * recharge_rate
                current_time += charge_time
                current_battery = battery_cap
                actual_seq.append(s)
                current_node = s
            # Now go to customer
            d = dist[current_node][c]
            total_dist += d
            current_time += d / speed
            current_battery -= d
            current_node = c
        else:
            total_dist += d
            current_time += d / speed
            current_battery -= d
            current_node = c

        # At customer
        if current_time > due[c]:
            return float('inf'), False
        current_time = max(current_time, ready[c])
        current_time += service[c]
        actual_seq.append(c)

    # Return to depot
    d_back = dist[current_node][0]
    if d_back > current_battery:
        # Need charging to get back
        station_seq = find_charging_stops(current_node, 0, current_battery, dist,
                                          stations, battery_cap)
        if station_seq is None:
            return float('inf'), False
        for s in station_seq:
            ds = dist[current_node][s]
            total_dist += ds
            current_time += ds / speed
            current_battery -= ds
            charge_time = (battery_cap - current_battery) * recharge_rate
            current_time += charge_time
            current_battery = battery_cap
            current_node = s
        d_back = dist[current_node][0]
        total_dist += d_back
        current_time += d_back / speed
        current_battery -= d_back
    else:
        total_dist += d_back
        current_time += d_back / speed
        current_battery -= d_back

    if current_time > horizon:
        return float('inf'), False
    if current_battery < 0:
        return float('inf'), False

    return total_dist, True


def find_charging_stops(from_node, to_node, current_battery, dist, stations, battery_cap):
    """
    Find a sequence of stations to insert between from_node and to_node
    such that battery never goes negative.
    Returns list of station nodes, or None if impossible.
    """
    # Simple approach: try direct, then try via each station
    d_direct = dist[from_node][to_node]
    if d_direct <= current_battery:
        return []  # No charging needed

    # Try via single station
    best_seq = None
    best_cost = float('inf')
    for s in stations:
        d1 = dist[from_node][s]
        d2 = dist[s][to_node]
        if d1 <= current_battery and d2 <= battery_cap:
            cost = d1 + d2
            if cost < best_cost:
                best_cost = cost
                best_seq = [s]

    if best_seq is not None:
        return best_seq

    # Try via two stations
    for s1 in stations:
        d1 = dist[from_node][s1]
        if d1 > current_battery:
            continue
        for s2 in stations:
            d2 = dist[s1][s2]
            d3 = dist[s2][to_node]
            if d2 <= battery_cap and d3 <= battery_cap:
                cost = d1 + d2 + d3
                if cost < best_cost:
                    best_cost = cost
                    best_seq = [s1, s2]

    return best_seq
# EVOLVE-BLOCK-END
