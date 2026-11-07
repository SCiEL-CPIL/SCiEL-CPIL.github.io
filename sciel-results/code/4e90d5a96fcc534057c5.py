"""EV routing with time windows, capacity, and battery constraints."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    t0 = time.time()
    DEADLINE = 140.0  # seconds, safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    depot = 0

    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Customer indices
    cust_ids = list(range(1, n_customers + 1))

    # Clarke-Wright savings initial solution
    # First, create individual routes for each customer
    routes = [[c] for c in cust_ids]

    # Compute savings: saving(i, j) = dist(depot, i) + dist(depot, j) - dist(i, j)
    savings = []
    for i in cust_ids:
        for j in cust_ids:
            if i != j:
                s = dist[depot, i] + dist[depot, j] - dist[i, j]
                savings.append((s, i, j))
    savings.sort(key=lambda x: -x[0])

    # Build route lookup
    def get_route_pos(node_id):
        for r_idx, route in enumerate(routes):
            if node_id in route:
                return r_idx, route.index(node_id)
        return -1, -1

    # Merge routes using savings
    for s_val, i, j in savings:
        if s_val <= 0:
            break
        ri, pi = get_route_pos(i)
        rj, pj = get_route_pos(j)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        # Check if i is at end of its route and j is at start of its route (or vice versa)
        route_i = routes[ri]
        route_j = routes[rj]
        merged = None
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_j[-1] == j and route_i[0] == i:
            merged = route_j + route_i
        if merged is None:
            continue
        # Check capacity
        total_demand = sum(demands[n] for n in merged)
        if total_demand > capacity:
            continue
        # Check battery feasibility (quick check without time windows)
        if not _quick_battery_check(merged, dist, depot, battery_max, recharge_rate, speed, stations, coords):
            continue
        routes[ri] = merged
        routes[rj] = []

    # Remove empty routes
    routes = [r for r in routes if r]

    # Now apply local search with full constraint checking
    def _check_route_valid(route, dist, depot, demands, ready, due, service,
                           battery_max, speed, recharge_rate, horizon, stations, coords):
        """Full feasibility check for a route."""
        if not route:
            return True
        total_demand = sum(demands[n] for n in route)
        if total_demand > capacity:
            return False

        t = 0.0
        battery = battery_max
        prev = depot
        for node in route:
            d = dist[prev, node]
            battery -= d
            if battery < -1e-9:
                return False
            t += d / speed
            if node in stations:
                # Recharge
                if battery < 0:
                    return False
                recharge_time = max(0, battery_max - battery) * recharge_rate
                t += recharge_time
                battery = battery_max
            else:
                # Customer
                t = max(t, ready[node])
                if t > due[node] + 1e-9:
                    return False
                t += service[node]
            prev = node
        # Return to depot
        d = dist[prev, depot]
        battery -= d
        if battery < -1e-9:
            return False
        t += d / speed
        if t > horizon + 1e-9:
            return False
        return True

    def _route_distance(route, dist, depot):
        """Total Euclidean distance of a route including depot legs."""
        if not route:
            return 0.0
        total = dist[depot, route[0]]
        for k in range(len(route) - 1):
            total += dist[route[k], route[k + 1]]
        total += dist[route[-1], depot]
        return total

    def _total_distance(routes, dist, depot):
        return sum(_route_distance(r, dist, depot) for r in routes)

    # Local search: 2-opt, Or-opt, cross-route moves
    best_dist = _total_distance(routes, dist, depot)
    no_improve = 0
    max_no_improve = 50

    def try_2opt(route_idx, routes, dist, depot):
        """Try 2-opt within a single route."""
        route = routes[route_idx]
        n = len(route)
        if n < 2:
            return False
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if new_route == route:
                    continue
                # Check capacity (same)
                # Check battery
                if not _quick_battery_check(new_route, dist, depot, battery_max, recharge_rate, speed, stations, coords):
                    continue
                # Full check
                if not _check_route_valid(new_route, dist, depot, demands, ready, due, service,
                                         battery_max, speed, recharge_rate, horizon, stations, coords):
                    continue
                old_d = _route_distance(route, dist, depot)
                new_d = _route_distance(new_route, dist, depot)
                if new_d < old_d - 1e-9:
                    routes[route_idx] = new_route
                    return True
        return False

    def try_or_opt(route_idx, routes, dist, depot):
        """Try Or-opt: move 1-3 consecutive nodes to another position in same route or another route."""
        route = routes[route_idx]
        n = len(route)
        if n < 2:
            return False
        improved = False
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                segment = route[i:i+seg_len]
                # Remove segment
                remaining = route[:i] + route[i+seg_len:]
                if not remaining:
                    continue
                # Try inserting at each position in remaining
                for pos in range(len(remaining) + 1):
                    new_route = remaining[:pos] + segment + remaining[pos:]
                    if new_route == route:
                        continue
                    if not _quick_battery_check(new_route, dist, depot, battery_max, recharge_rate, speed, stations, coords):
                        continue
                    if not _check_route_valid(new_route, dist, depot, demands, ready, due, service,
                                             battery_max, speed, recharge_rate, horizon, stations, coords):
                        continue
                    old_d = _route_distance(route, dist, depot)
                    new_d = _route_distance(new_route, dist, depot)
                    if new_d < old_d - 1e-9:
                        routes[route_idx] = new_route
                        return True
                # Try moving segment to another route
                for other_idx in range(len(routes)):
                    if other_idx == route_idx:
                        continue
                    other_route = routes[other_idx]
                    for pos in range(len(other_route) + 1):
                        new_other = other_route[:pos] + segment + other_route[pos:]
                        if not _quick_battery_check(new_other, dist, depot, battery_max, recharge_rate, speed, stations, coords):
                            continue
                        if not _check_route_valid(new_other, dist, depot, demands, ready, due, service,
                                                 battery_max, speed, recharge_rate, horizon, stations, coords):
                            continue
                        # Check remaining route is still valid
                        if remaining and not _check_route_valid(remaining, dist, depot, demands, ready, due, service,
                                                               battery_max, speed, recharge_rate, horizon, stations, coords):
                            continue
                        old_total = _route_distance(route, dist, depot) + _route_distance(other_route, dist, depot)
                        new_total = _route_distance(remaining, dist, depot) + _route_distance(new_other, dist, depot)
                        if new_total < old_total - 1e-9:
                            routes[route_idx] = remaining
                            routes[other_idx] = new_other
                            return True
        return improved

    def try_cross_route_swap(routes, dist, depot):
        """Try swapping a node from one route with a node from another."""
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                route_i = routes[i]
                route_j = routes[j]
                for pi in range(len(route_i)):
                    for pj in range(len(route_j)):
                        a = route_i[pi]
                        b = route_j[pj]
                        new_i = route_i[:pi] + [b] + route_i[pi+1:]
                        new_j = route_j[:pj] + [a] + route_j[pj+1:]
                        if not _quick_battery_check(new_i, dist, depot, battery_max, recharge_rate, speed, stations, coords):
                            continue
                        if not _quick_battery_check(new_j, dist, depot, battery_max, recharge_rate, speed, stations, coords):
                            continue
                        if not _check_route_valid(new_i, dist, depot, demands, ready, due, service,
                                                 battery_max, speed, recharge_rate, horizon, stations, coords):
                            continue
                        if not _check_route_valid(new_j, dist, depot, demands, ready, due, service,
                                                 battery_max, speed, recharge_rate, horizon, stations, coords):
                            continue
                        old_total = _route_distance(route_i, dist, depot) + _route_distance(route_j, dist, depot)
                        new_total = _route_distance(new_i, dist, depot) + _route_distance(new_j, dist, depot)
                        if new_total < old_total - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
                            return True
        return False

    def _quick_battery_check(route, dist, depot, battery_max, recharge_rate, speed, stations, coords):
        """Quick battery feasibility (ignores time windows)."""
        if not route:
            return True
        battery = battery_max
        prev = depot
        for node in route:
            d = dist[prev, node]
            battery -= d
            if battery < -1e-9:
                return False
            if node in stations:
                battery = battery_max
            prev = node
        d = dist[prev, depot]
        battery -= d
        if battery < -1e-9:
            return False
        return True

    # Run local search until no improvement or deadline
    while time.time() < DEADLINE:
        improved = False
        # 2-opt on each route
        for ri in range(len(routes)):
            if time.time() > DEADLINE:
                break
            if try_2opt(ri, routes, dist, depot):
                improved = True
        # Or-opt
        if not improved:
            for ri in range(len(routes)):
                if time.time() > DEADLINE:
                    break
                if try_or_opt(ri, routes, dist, depot):
                    improved = True
        # Cross-route swap
        if not improved:
            if try_cross_route_swap(routes, dist, depot):
                improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                break

    # Final validation
    valid_routes = []
    for r in routes:
        if _check_route_valid(r, dist, depot, demands, ready, due, service,
                             battery_max, speed, recharge_rate, horizon, stations, coords):
            valid_routes.append(r)
        else:
            # Split into individual customer routes
            for c in r:
                valid_routes.append([c])

    return valid_routes
# EVOLVE-BLOCK-END
