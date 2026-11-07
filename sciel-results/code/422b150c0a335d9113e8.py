"""EV routing: Savings construction with auto-station insertion + inter-route Or-Opt + 2-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import heapq

DEADLINE_SECONDS = 115.0


def _dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _validate_route(route, coords, demands, ready, due, service, capacity,
                    battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    """Validate a single route. Returns (valid, distance)."""
    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0

    cur = 0
    t = 0.0
    battery = battery_max
    dist = 0.0

    for node in route:
        d = _dist(coords[cur], coords[node])
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return False, 0.0
        dist += d
        cur = node

        if node in stations_set:
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, 0.0
            t += service[node]

    d = _dist(coords[cur], coords[0])
    battery -= d
    t += d / speed
    dist += d
    if battery < -1e-9:
        return False, 0.0
    if t > horizon + 1e-9:
        return False, 0.0

    return True, dist


def _compute_distance(route, coords):
    if not route:
        return 0.0
    cur = 0
    total = 0.0
    for node in route:
        total += _dist(coords[cur], coords[node])
        cur = node
    total += _dist(coords[cur], coords[0])
    return total


def _try_route_with_stations(order, coords, demands, ready, due, service,
                             battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, capacity):
    """Given an order of customers, try inserting stations to make it feasible.
    Returns (route, dist) or (None, inf)."""
    total_demand = sum(demands[c] for c in order)
    if total_demand > capacity:
        return None, float('inf')

    # Simulate the route, inserting stations when battery would go negative
    route = []
    cur = 0
    t = 0.0
    battery = battery_max
    dist = 0.0

    for node in order:
        d = _dist(coords[cur], coords[node])
        if battery - d < -1e-9:
            # Need to charge: find best station to insert before this node
            best_station = None
            best_extra = float('inf')
            for st in stations_set:
                d1 = _dist(coords[cur], coords[st])
                d2 = _dist(coords[st], coords[node])
                if battery - d1 < -1e-9:
                    continue
                extra = d1 + d2 - d
                extra_time = d1 / speed + (battery_max - (battery - d1)) * recharge_rate + d2 / speed
                if extra < best_extra:
                    best_extra = extra
                    best_station = st
                    best_extra_time = extra_time
            if best_station is None:
                return None, float('inf')
            # Insert station
            d1 = _dist(coords[cur], coords[best_station])
            battery -= d1
            t += d1 / speed
            dist += d1
            cur = best_station
            route.append(cur)
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
            # Now go to node
            d = _dist(coords[cur], coords[node])
            battery -= d
            t += d / speed
            dist += d
            cur = node
        else:
            battery -= d
            t += d / speed
            dist += d
            cur = node

        if cur in stations_set:
            t += (battery_max - battery) * recharge_rate
            battery = battery_max
        else:
            t = max(t, ready[cur])
            if t > due[cur] + 1e-9:
                return None, float('inf')
            t += service[cur]
            route.append(cur)

    # Return to depot
    d = _dist(coords[cur], coords[0])
    if battery - d < -1e-9:
        # Try inserting station before return
        best_station = None
        best_extra = float('inf')
        for st in stations_set:
            d1 = _dist(coords[cur], coords[st])
            d2 = _dist(coords[st], coords[0])
            if battery - d1 < -1e-9:
                continue
            extra = d1 + d2 - d
            if extra < best_extra:
                best_extra = extra
                best_station = st
        if best_station is None:
            return None, float('inf')
        d1 = _dist(coords[cur], coords[best_station])
        battery -= d1
        t += d1 / speed
        dist += d1
        cur = best_station
        route.append(cur)
        t += (battery_max - battery) * recharge_rate
        battery = battery_max
        d = _dist(coords[cur], coords[0])
        battery -= d
        t += d / speed
        dist += d
    else:
        battery -= d
        t += d / speed
        dist += d

    if t > horizon + 1e-9:
        return None, float('inf')

    return route, dist


def _savings_construction(customers, coords, demands, ready, due, service,
                          capacity, battery_max, speed, recharge_rate, horizon,
                          stations_set, n_customers):
    """Clarke-Wright savings construction with auto-station insertion."""
    n = len(customers)
    if n == 0:
        return []

    # Compute depot distances
    depot_d = np.array([_dist(coords[0], coords[c]) for c in customers])

    # Compute savings
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            s = depot_d[i] + depot_d[j] - _dist(coords[customers[i]], coords[customers[j]])
            savings.append((-s, i, j))
    savings.sort()  # Most negative first = largest savings

    # Initialize: each customer is its own route
    routes = [[customers[i]] for i in range(n)]
    route_of = list(range(n))  # which route index each customer is in
    route_nodes = {i: [customers[i]] for i in range(n)}
    route_demand = {i: demands[customers[i]] for i in range(n)}
    route_start = list(customers)  # start node of each route (for savings)
    route_end = list(customers)    # end node of each route

    # Track which routes are "open" (can be merged)
    active = set(range(n))

    for neg_s, i, j in savings:
        if time.time() > DEADLINE_SECONDS - 5:
            break
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        if ri not in active or rj not in active:
            continue

        # Check if i is at the end of route ri and j is at the start of route rj
        # Or i at start and j at end, etc.
        # In savings, we merge: ...i | j... -> ...i, j...
        # i must be the last node of route ri, j must be the first node of route rj
        # OR i first of ri, j last of rj (reverse)

        # Option 1: i is end of ri, j is start of rj
        # Option 2: j is end of rj, i is start of ri (reverse merge)

        # Try option 1: append rj after ri (i at end of ri, j at start of rj)
        # Actually in savings: route ri ends with i, route rj starts with j
        # Merged: ri + rj (i adjacent to j)
        # Or ri + rj reversed

        # Let's try both directions
        best_merge = None
        best_dist = float('inf')

        # Direction 1: ri (ending at i) + rj (starting at j)
        if route_nodes[ri][-1] == customers[i] and route_nodes[rj][0] == customers[j]:
            combined_order = route_nodes[ri] + route_nodes[rj]
            total_dem = route_demand[ri] + route_demand[rj]
            if total_dem <= capacity:
                route, d = _try_route_with_stations(combined_order, coords, demands, ready, due,
                                                     service, battery_max, speed, recharge_rate,
                                                     horizon, stations_set, n_customers, capacity)
                if route is not None and d < best_dist:
                    best_dist = d
                    best_merge = (ri, rj, combined_order, route)

        # Direction 2: rj (ending at j) + ri (starting at i)
        if route_nodes[rj][-1] == customers[j] and route_nodes[ri][0] == customers[i]:
            combined_order = route_nodes[rj] + route_nodes[ri]
            total_dem = route_demand[ri] + route_demand[rj]
            if total_dem <= capacity:
                route, d = _try_route_with_stations(combined_order, coords, demands, ready, due,
                                                     service, battery_max, speed, recharge_rate,
                                                     horizon, stations_set, n_customers, capacity)
                if route is not None and d < best_dist:
                    best_dist = d
                    best_merge = (rj, ri, combined_order, route)

        # Direction 3: ri (ending at i) + reversed rj (ending at j)
        if route_nodes[ri][-1] == customers[i] and route_nodes[rj][-1] == customers[j]:
            combined_order = route_nodes[ri] + list(reversed(route_nodes[rj]))
            total_dem = route_demand[ri] + route_demand[rj]
            if total_dem <= capacity:
                route, d = _try_route_with_stations(combined_order, coords, demands, ready, due,
                                                     service, battery_max, speed, recharge_rate,
                                                     horizon, stations_set, n_customers, capacity)
                if route is not None and d < best_dist:
                    best_dist = d
                    best_merge = (ri, rj, combined_order, route)

        # Direction 4: rj (ending at j) + reversed ri (ending at i)
        if route_nodes[rj][-1] == customers[j] and route_nodes[ri][-1] == customers[i]:
            combined_order = route_nodes[rj] + list(reversed(route_nodes[ri]))
            total_dem = route_demand[ri] + route_demand[rj]
            if total_dem <= capacity:
                route, d = _try_route_with_stations(combined_order, coords, demands, ready, due,
                                                     service, battery_max, speed, recharge_rate,
                                                     horizon, stations_set, n_customers, capacity)
                if route is not None and d < best_dist:
                    best_dist = d
                    best_merge = (rj, ri, combined_order, route)

        if best_merge is not None:
            ri2, rj2, order, route = best_merge
            # Remove rj2 from active, update ri2
            active.discard(rj2)
            active.discard(ri2)
            # New route index
            new_idx = max(active) + 1 if active else 0
            active.add(new_idx)
            route_nodes[new_idx] = order
            route_demand[new_idx] = route_demand[ri2] + route_demand[rj2]
            # Update route_of for all customers in both routes
            for c in order:
                ci = customers.index(c)
                route_of[ci] = new_idx
            # Clean up
            del route_nodes[ri2]
            del route_nodes[rj2]
            del route_demand[ri2]
            del route_demand[rj2]

    # Convert to final routes
    final_routes = []
    for idx in sorted(route_nodes.keys()):
        order = route_nodes[idx]
        route, d = _try_route_with_stations(order, coords, demands, ready, due,
                                            service, battery_max, speed, recharge_rate,
                                            horizon, stations_set, n_customers, capacity)
        if route is not None:
            final_routes.append(route)
        else:
            # Fallback: individual customers
            for c in order:
                single, sd = _try_route_with_stations([c], coords, demands, ready, due,
                                                      service, battery_max, speed, recharge_rate,
                                                      horizon, stations_set, n_customers, capacity)
                if single is not None:
                    final_routes.append(single)
                else:
                    final_routes.append([c])

    return final_routes


def _two_opt(route, coords, demands, ready, due, service, capacity,
             battery_max, speed, recharge_rate, horizon, stations_set, n_customers,
             max_iters=10):
    """2-opt with fast feasibility check."""
    if len(route) < 2:
        return route

    best_route = list(route)
    best_dist = _compute_distance(best_route, coords)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(best_route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                valid, d = _validate_route(candidate, coords, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
                if valid and d < best_dist - 1e-9:
                    best_route = candidate
                    best_dist = d
                    improved = True

    return best_route


def _or_opt(route, coords, demands, ready, due, service, capacity,
            battery_max, speed, recharge_rate, horizon, stations_set, n_customers,
            max_iters=8):
    """Or-opt: move 1-2 consecutive nodes."""
    if len(route) < 4:
        return route

    best_route = list(route)
    best_dist = _compute_distance(best_route, coords)
    improved = True
    iters = 0

    while improved and iters < max_iters:
        improved = False
        iters += 1
        n = len(best_route)
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                segment = best_route[i:i + seg_len]
                rest = best_route[:i] + best_route[i + seg_len:]
                for j in range(len(rest) + 1):
                    if i > 0 and j <= i:
                        continue
                    if i + seg_len <= j:
                        continue
                    candidate = rest[:j] + segment + rest[j:]
                    valid, d = _validate_route(candidate, coords, demands, ready, due, service,
                                               capacity, battery_max, speed, recharge_rate,
                                               horizon, stations_set, n_customers)
                    if valid and d < best_dist - 1e-9:
                        best_route = candidate
                        best_dist = d
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route


def _inter_route_move(routes, coords, demands, ready, due, service, capacity,
                      battery_max, speed, recharge_rate, horizon, stations_set, n_customers,
                      max_moves=40):
    """Move a customer from one route to another if it reduces total distance."""
    improved = True
    moves = 0
    while improved and moves < max_moves:
        improved = False
        moves += 1
        for i in range(len(routes)):
            if len(routes[i]) <= 1:
                continue
            for j in range(len(routes[i])):
                cust = routes[i][j]
                if not (1 <= cust <= n_customers):
                    continue
                for k in range(len(routes)):
                    if k == i:
                        continue
                    # Try inserting cust at each position in route k
                    for pos in range(len(routes[k]) + 1):
                        new_i = routes[i][:j] + routes[i][j + 1:]
                        new_k = routes[k][:pos] + [cust] + routes[k][pos:]

                        # Quick capacity check
                        demand_k = sum(demands[v] for v in new_k if 1 <= v <= n_customers)
                        if demand_k > capacity:
                            continue

                        if new_i:
                            valid_i, d_i = _validate_route(new_i, coords, demands, ready, due, service,
                                                           capacity, battery_max, speed, recharge_rate,
                                                           horizon, stations_set, n_customers)
                        else:
                            valid_i, d_i = True, 0.0

                        valid_k, d_k = _validate_route(new_k, coords, demands, ready, due, service,
                                                       capacity, battery_max, speed, recharge_rate,
                                                       horizon, stations_set, n_customers)

                        if valid_i and valid_k:
                            old_d = _compute_distance(routes[i], coords) + _compute_distance(routes[k], coords)
                            new_d = d_i + d_k
                            if new_d < old_d - 1e-9:
                                routes[i] = new_i
                                routes[k] = new_k
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    routes = [r for r in routes if r]
    return routes


def _try_insert_station(route, coords, demands, ready, due, service, capacity,
                        battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    """Try inserting a station at best position if it reduces distance (shouldn't happen) 
    or makes route feasible."""
    best_route = list(route)
    best_dist = _compute_distance(best_route, coords)

    for station in stations_set:
        for pos in range(len(best_route) + 1):
            candidate = best_route[:pos] + [station] + best_route[pos:]
            valid, d = _validate_route(candidate, coords, demands, ready, due, service,
                                       capacity, battery_max, speed, recharge_rate,
                                       horizon, stations_set, n_customers)
            if valid and d < best_dist - 1e-9:
                best_route = candidate
                best_dist = d

    return best_route


def _validate_solution(routes, coords, demands, ready, due, service, capacity,
                       battery_max, speed, recharge_rate, horizon, stations_set, n_customers):
    all_visited = []
    for route in routes:
        for node in route:
            if 1 <= node <= n_customers:
                all_visited.append(node)
    if len(all_visited) != n_customers:
        return False, 0.0
    if len(set(all_visited)) != n_customers:
        return False, 0.0

    total_dist = 0.0
    for route in routes:
        valid, d = _validate_route(route, coords, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate,
                                   horizon, stations_set, n_customers)
        if not valid:
            return False, 0.0
        total_dist += d
    return True, total_dist


def plan(instance):
    start_time = time.time()
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = set(instance["stations"])
    n_customers = instance["n_customers"]

    customers = list(range(1, n_customers + 1))

    # Savings-based construction
    routes = _savings_construction(customers, coords, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate, horizon,
                                   stations_set, n_customers)

    # Check coverage
    covered = set()
    for r in routes:
        for node in r:
            if 1 <= node <= n_customers:
                covered.add(node)
    missing = [c for c in customers if c not in covered]
    for c in missing:
        routes.append([c])

    # Validate
    valid, total_dist = _validate_solution(routes, coords, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
    if not valid:
        # Fallback: one customer per route
        return [[v] for v in customers]

    # Intra-route optimization: 2-opt + or-opt
    for i in range(len(routes)):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        routes[i] = _two_opt(routes[i], coords, demands, ready, due, service,
                             capacity, battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, max_iters=10)
        routes[i] = _or_opt(routes[i], coords, demands, ready, due, service,
                            capacity, battery_max, speed, recharge_rate, horizon,
                            stations_set, n_customers, max_iters=6)

    # Inter-route moves
    if time.time() - start_time < DEADLINE_SECONDS - 10:
        routes = _inter_route_move(routes, coords, demands, ready, due, service,
                                   capacity, battery_max, speed, recharge_rate, horizon,
                                   stations_set, n_customers, max_moves=40)

    # Another round of intra-route optimization after inter-route moves
    for i in range(len(routes)):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        routes[i] = _two_opt(routes[i], coords, demands, ready, due, service,
                             capacity, battery_max, speed, recharge_rate, horizon,
                             stations_set, n_customers, max_iters=6)
        routes[i] = _or_opt(routes[i], coords, demands, ready, due, service,
                            capacity, battery_max, speed, recharge_rate, horizon,
                            stations_set, n_customers, max_iters=4)

    # Final validation
    valid, total_dist = _validate_solution(routes, coords, demands, ready, due, service,
                                           capacity, battery_max, speed, recharge_rate,
                                           horizon, stations_set, n_customers)
    if not valid:
        return [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
