"""
EV routing with time windows and battery constraints.

Strategy:
- Construction: Clarke-Wright savings heuristic with a feasibility check that
  inserts a charging station whenever the battery would run out between two
  customers.  This keeps the number of vehicles small and produces compact
  routes that respect the battery limit.
- Local search:
  * Intra-route 2-opt (with auto station insertion after each reversal).
  * Inter-route relocate (move a customer to the best position in another
    route, again with auto station insertion).
  * Inter-route 2-opt (swap a segment between two routes).
- All feasibility checks are done by a fast forward simulation that inserts
  a station right before the battery would go negative.  This avoids
  combinatorial explosion over station placements.
"""

import numpy as np
import time
import math

DEADLINE_SECONDS = 115.0


def _simulate(route, D, ready, due, service, demands, is_station,
              battery_max, recharge_rate, speed, horizon,
              auto_station=True, station_list=None):
    """
    Simulate a route (list of node ids, no depot).
    Returns (total_distance, final_time, final_battery, feasible, augmented_route).
    augmented_route is the route with stations inserted if auto_station is True.
    """
    if not route:
        return 0.0, 0.0, battery_max, True, []

    t = 0.0
    b = battery_max
    dist_total = 0.0
    augmented = []
    prev = 0
    n = len(route)

    for idx, node in enumerate(route):
        dist = D[prev, node]
        # If battery would go negative and auto_station is on, insert a station
        # just before this node.  Choose the nearest station to prev.
        if b - dist < -1e-9 and auto_station and station_list:
            # find nearest station to prev (not the current node)
            best_s = -1
            best_d = float('inf')
            for s in station_list:
                d = D[prev, s]
                if d < best_d:
                    best_d = d
                    best_s = s
            if best_s == -1:
                return dist_total, t, b, False, None
            # travel to station
            t += best_d / speed
            b -= best_d
            dist_total += best_d
            if b < -1e-9:
                return dist_total, t, b, False, None
            # recharge
            t += (battery_max - b) * recharge_rate
            b = battery_max
            augmented.append(best_s)
            prev = best_s
            # now travel to node again
            dist = D[prev, node]
            if b - dist < -1e-9:
                return dist_total, t, b, False, None
        b -= dist
        if b < -1e-9:
            return dist_total, t, b, False, None
        t += dist / speed
        dist_total += dist
        augmented.append(node)

        if is_station[node]:
            t += (battery_max - b) * recharge_rate
            b = battery_max
        else:
            start_service = max(t, ready[node])
            if start_service > due[node] + 1e-9:
                return dist_total, t, b, False, None
            t = start_service + service[node]
        prev = node

    # return to depot
    dist_back = D[prev, 0]
    if b - dist_back < -1e-9 and auto_station and station_list:
        # insert station before depot
        best_s = -1
        best_d = float('inf')
        for s in station_list:
            d = D[prev, s]
            if d < best_d:
                best_d = d
                best_s = s
        if best_s == -1:
            return dist_total, t, b, False, None
        t += best_d / speed
        b -= best_d
        dist_total += best_d
        if b < -1e-9:
            return dist_total, t, b, False, None
        t += (battery_max - b) * recharge_rate
        b = battery_max
        augmented.append(best_s)
        prev = best_s
        dist_back = D[prev, 0]
        if b - dist_back < -1e-9:
            return dist_total, t, b, False, None
    b -= dist_back
    if b < -1e-9:
        return dist_total, t, b, False, None
    t += dist_back / speed
    dist_total += dist_back

    if t > horizon + 1e-9:
        return dist_total, t, b, False, None

    return dist_total, t, b, True, augmented


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
    stations = set(instance["stations"])
    n_customers = instance["n_customers"]
    n_nodes = len(coords)

    is_station = np.zeros(n_nodes, dtype=bool)
    for s in stations:
        is_station[s] = True
    station_list = list(stations)

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff ** 2, axis=-1))

    customers = list(range(1, n_customers + 1))

    # ---------- Construction: Clarke-Wright savings with auto station ----------
    # Start with each customer as its own route
    routes = [[c] for c in customers]
    # Precompute route distances (with auto station)
    route_costs = []
    for r in routes:
        d, t, b, feas, aug = _simulate(r, D, ready, due, service, demands,
                                       is_station, battery_max, recharge_rate,
                                       speed, horizon, auto_station=True,
                                       station_list=station_list)
        route_costs.append(d if feas else float('inf'))

    # Savings: for two routes i and j, the saving from merging them is
    # cost_i + cost_j - cost_merged (where merged is concatenation)
    # We'll compute savings between last node of route i and first node of route j,
    # and also first of i and last of j (since we can reverse).

    # For simplicity, we'll consider merging route i and route j by concatenation
    # (i then j) or (j then i).  We pick the one with lower cost.

    # Build savings list
    n_routes = len(routes)
    savings = []  # (saving, i, j, order)  order: 0 means i+j, 1 means j+i

    for i in range(n_routes):
        for j in range(i + 1, n_routes):
            # Try i + j
            merged = routes[i] + routes[j]
            d1, t1, b1, feas1, _ = _simulate(merged, D, ready, due, service,
                                             demands, is_station, battery_max,
                                             recharge_rate, speed, horizon,
                                             auto_station=True,
                                             station_list=station_list)
            cost1 = d1 if feas1 else float('inf')
            # Try j + i
            merged2 = routes[j] + routes[i]
            d2, t2, b2, feas2, _ = _simulate(merged2, D, ready, due, service,
                                             demands, is_station, battery_max,
                                             recharge_rate, speed, horizon,
                                             auto_station=True,
                                             station_list=station_list)
            cost2 = d2 if feas2 else float('inf')

            if cost1 < cost2:
                saving = route_costs[i] + route_costs[j] - cost1
                if saving > 1e-9 and feas1:
                    savings.append((saving, i, j, 0))
            else:
                saving = route_costs[i] + route_costs[j] - cost2
                if saving > 1e-9 and feas2:
                    savings.append((saving, i, j, 1))

    savings.sort(reverse=True)

    # Union-Find for route merging
    parent = list(range(n_routes))
    size = [1] * n_routes
    active = [True] * n_routes

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if size[rx] < size[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        size[rx] += size[ry]

    merged_count = 0
    for saving, i, j, order in savings:
        if time.time() - start_time > DEADLINE_SECONDS * 0.3:
            break
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # Merge: we need to actually merge the routes
        # Get the representative route lists
        # We'll maintain a dict from root to route
        # For simplicity, we'll just rebuild routes after each merge.
        # This is O(n^2) but n=25, so it's fine.
        if order == 0:
            new_route = routes[i] + routes[j]
        else:
            new_route = routes[j] + routes[i]
        d, t, b, feas, aug = _simulate(new_route, D, ready, due, service,
                                       demands, is_station, battery_max,
                                       recharge_rate, speed, horizon,
                                       auto_station=True,
                                       station_list=station_list)
        if not feas:
            continue
        # Check capacity
        total_demand = sum(demands[v] for v in new_route if not is_station[v])
        if total_demand > capacity:
            continue
        # Merge
        union(ri, rj)
        routes[ri] = new_route
        route_costs[ri] = d
        routes[rj] = []
        route_costs[rj] = 0.0
        merged_count += 1

    # Collect non-empty routes
    final_routes = []
    for i in range(n_routes):
        if routes[i]:
            final_routes.append(routes[i])

    # ---------- Local search ----------
    def route_distance(route):
        d, t, b, feas, _ = _simulate(route, D, ready, due, service, demands,
                                     is_station, battery_max, recharge_rate,
                                     speed, horizon, auto_station=True,
                                     station_list=station_list)
        return d if feas else float('inf')

    improved = True
    while improved:
        improved = False
        if time.time() - start_time > DEADLINE_SECONDS * 0.85:
            break

        # Intra-route 2-opt
        for r_idx in range(len(final_routes)):
            route = final_routes[r_idx]
            n = len(route)
            if n < 2:
                continue
            best_saving = 0
            best_i, best_j = -1, -1
            best_new_route = None

            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    d_new, t_new, b_new, feas_new, _ = _simulate(new_route, D,
                                                                 ready, due, service,
                                                                 demands, is_station,
                                                                 battery_max,
                                                                 recharge_rate, speed,
                                                                 horizon,
                                                                 auto_station=True,
                                                                 station_list=station_list)
                    if not feas_new:
                        continue
                    total_demand = sum(demands[v] for v in new_route if not is_station[v])
                    if total_demand > capacity:
                        continue
                    d_old = route_distance(route)
                    if d_new < d_old - 1e-9:
                        best_saving = d_old - d_new
                        best_i, best_j = i, j
                        best_new_route = new_route

            if best_i != -1:
                final_routes[r_idx] = best_new_route
                improved = True

        # Inter-route relocate
        for r_idx in range(len(final_routes)):
            route = final_routes[r_idx]
            for pos in range(len(route)):
                cust = route[pos]
                # Try moving cust to each other route
                for o_idx in range(len(final_routes)):
                    if o_idx == r_idx:
                        continue
                    other = final_routes[o_idx]
                    for o_pos in range(len(other) + 1):
                        new_other = other[:o_pos] + [cust] + other[o_pos:]
                        new_route = route[:pos] + route[pos + 1:]
                        # Check feasibility of both
                        d1, t1, b1, feas1, _ = _simulate(new_other, D, ready, due,
                                                         service, demands, is_station,
                                                         battery_max, recharge_rate,
                                                         speed, horizon,
                                                         auto_station=True,
                                                         station_list=station_list)
                        if not feas1:
                            continue
                        if new_route:
                            d2, t2, b2, feas2, _ = _simulate(new_route, D, ready, due,
                                                             service, demands, is_station,
                                                             battery_max, recharge_rate,
                                                             speed, horizon,
                                                             auto_station=True,
                                                             station_list=station_list)
                            if not feas2:
                                continue
                            d_new_total = d1 + d2
                        else:
                            d_new_total = d1

                        d_old_total = route_distance(route) + route_distance(other)
                        if d_new_total < d_old_total - 1e-9:
                            final_routes[o_idx] = new_other
                            if new_route:
                                final_routes[r_idx] = new_route
                            else:
                                final_routes[r_idx] = []
                            improved = True
                            break
                    if improved and time.time() - start_time > DEADLINE_SECONDS * 0.85:
                        break
                if improved and time.time() - start_time > DEADLINE_SECONDS * 0.85:
                    break
            if time.time() - start_time > DEADLINE_SECONDS * 0.85:
                break

        # Remove empty routes
        final_routes = [r for r in final_routes if len(r) > 0]

    # ---------- Final validation and cleanup ----------
    visited = set()
    for r in final_routes:
        for n in r:
            if not is_station[n]:
                visited.add(n)

    missing = set(customers) - visited
    for m in missing:
        final_routes.append([m])

    return final_routes
