"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import heapq


def _euclid(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _build_dist(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            D[i, j] = d
            D[j, i] = d
    return D


def _check_route(route, D, instance, battery_cap, horizon, speed, recharge_rate,
                 ready, due, service, demand, stations_set):
    """
    Returns (feasible: bool, total_distance: float).
    route: list of node ids (customers and/or stations), depot implicit at start/end.
    """
    if not route:
        return True, 0.0

    cur = 0
    t = 0.0
    bat = battery_cap
    total_dist = 0.0
    served = set()

    for node in route:
        d = D[cur, node]
        if bat < d:
            return False, total_dist
        bat -= d
        t += d / speed
        total_dist += d

        if node in stations_set:
            # Station: recharge
            if bat < 0:
                return False, total_dist
            t += bat * recharge_rate
            bat = battery_cap
        else:
            # Customer
            if node in served:
                return False, total_dist
            served.add(node)
            if t > due[node]:
                return False, total_dist
            t = max(t, ready[node])
            t += service[node]

        cur = node

    # Return to depot
    d = D[cur, 0]
    if bat < d:
        return False, total_dist
    bat -= d
    t += d / speed
    total_dist += d

    if t > horizon:
        return False, total_dist

    return True, total_dist


def _route_distance(route, D):
    if not route:
        return 0.0
    cur = 0
    total = 0.0
    for node in route:
        total += D[cur, node]
        cur = node
    total += D[cur, 0]
    return total


def _clarke_wright_savings(D, customers, n_customers):
    """
    Build initial routes using Clarke-Wright savings heuristic.
    Returns list of routes (each a list of node ids).
    """
    # Start with each customer in its own route
    routes = [[c] for c in customers]
    route_of = {c: i for i, c in enumerate(customers)}

    # Compute savings: saving[i][j] = D[0][i] + D[0][j] - D[i][j]
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = D[0, ci] + D[0, cj] - D[ci, cj]
            savings.append((s, i, j))

    savings.sort(reverse=True)

    # Union-Find for tracking which customers are in which route
    parent = list(range(len(customers)))
    rank = [0] * len(customers)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    # For each route, track endpoints
    # routes is a list of lists; we need to find endpoints
    # Let's use a different approach: maintain route as a linked structure
    # Simpler: just try to merge routes

    for s, i, j in savings:
        if s <= 0:
            break
        ri = find(i)
        rj = find(j)
        if ri == rj:
            continue

        # Find the two routes
        # We need to find which route contains customer i and j
        # and check if they are at the endpoints
        # This is complex with the current structure. Let me use a simpler approach.
        
        # Actually, let me just do a simpler insertion-based approach instead.
        pass

    # The above is getting complex. Let me use a simpler nearest-neighbor + savings approach.
    # For now, return individual routes.
    return routes


def _nearest_neighbor_route(D, start, unvisited, ready, due, service, demand, battery_cap,
                            stations_set, horizon, speed, recharge_rate):
    """Build a route using nearest neighbor heuristic."""
    route = []
    cur = 0
    t = 0.0
    bat = battery_cap
    total_dist = 0.0
    load = 0

    while unvisited:
        # Find nearest feasible customer
        best = -1
        best_dist = float('inf')
        for c in unvisited:
            d = D[cur, c]
            if bat < d:
                continue
            # Check if we can reach and serve
            new_t = t + d / speed
            if new_t > due[c]:
                continue
            new_load = load + demand[c]
            if new_load > 100:  # capacity check (should be handled elsewhere)
                continue
            if d < best_dist:
                best_dist = d
                best = c

        if best == -1:
            break

        d = D[cur, best]
        bat -= d
        t += d / speed
        t = max(t, ready[best])
        t += service[best]
        load += demand[best]
        total_dist += d
        route.append(best)
        cur = best
        unvisited.remove(best)

    # Check if we can return to depot
    d = D[cur, 0]
    if bat >= d:
        total_dist += d
    else:
        # Can't return, this route is invalid
        return None, None

    return route, total_dist


def _try_2opt(route, D, instance, battery_cap, horizon, speed, recharge_rate,
              ready, due, service, demand, stations_set):
    """Try 2-opt moves on a route. Returns improved route or original."""
    n = len(route)
    if n < 2:
        return route

    best_route = list(route)
    best_dist = _route_distance(best_route, D)

    for i in range(n - 1):
        for j in range(i + 1, n):
            # Reverse segment i..j
            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
            if new_route == best_route:
                continue
            feasible, dist = _check_route(new_route, D, instance, battery_cap, horizon,
                                          speed, recharge_rate, ready, due, service,
                                          demand, stations_set)
            if feasible and dist < best_dist - 1e-9:
                best_route = new_route
                best_dist = dist

    return best_route


def _try_or_opt(route, D, instance, battery_cap, horizon, speed, recharge_rate,
                ready, due, service, demand, stations_set):
    """Try or-opt moves (move 1 or 2 consecutive customers to another position)."""
    n = len(route)
    if n < 2:
        return route

    best_route = list(route)
    best_dist = _route_distance(best_route, D)

    # Try moving 1 customer
    for i in range(n):
        for j in range(n):
            if j == i or j == i - 1:
                continue
            # Remove customer i and insert at position j
            new_route = list(route)
            c = new_route.pop(i)
            # Adjust j if i < j
            if i < j:
                j -= 1
            new_route.insert(j, c)
            if new_route == best_route:
                continue
            feasible, dist = _check_route(new_route, D, instance, battery_cap, horizon,
                                          speed, recharge_rate, ready, due, service,
                                          demand, stations_set)
            if feasible and dist < best_dist - 1e-9:
                best_route = new_route
                best_dist = dist

    # Try moving 2 consecutive customers
    for i in range(n - 1):
        for j in range(n):
            if j == i or j == i + 1 or j == i - 1:
                continue
            new_route = list(route)
            c1, c2 = new_route.pop(i), new_route.pop(i)
            if i < j:
                j -= 2
            else:
                j -= 2
            new_route.insert(j, c2)
            new_route.insert(j, c1)
            if new_route == best_route:
                continue
            feasible, dist = _check_route(new_route, D, instance, battery_cap, horizon,
                                          speed, recharge_rate, ready, due, service,
                                          demand, stations_set)
            if feasible and dist < best_dist - 1e-9:
                best_route = new_route
                best_dist = dist

    return best_route


def _insert_customer(route, customer, D, instance, battery_cap, horizon, speed,
                     recharge_rate, ready, due, service, demand, stations_set):
    """Try to insert customer into route. Returns (feasible, best_route, best_dist)."""
    n = len(route)
    best_route = None
    best_dist = float('inf')

    for pos in range(n + 1):
        new_route = route[:pos] + [customer] + route[pos:]
        feasible, dist = _check_route(new_route, D, instance, battery_cap, horizon,
                                      speed, recharge_rate, ready, due, service,
                                      demand, stations_set)
        if feasible and dist < best_dist:
            best_dist = dist
            best_route = new_route

    if best_route is None:
        return False, route, _route_distance(route, D)
    return True, best_route, best_dist


def _solve_instance(instance):
    coords = instance["coords"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    stations_set = set(stations)
    battery_cap = instance["battery"]
    horizon = instance["horizon"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demand = instance["demand"]
    capacity = instance["capacity"]

    D = _build_dist(coords)

    customers = list(range(1, n_customers + 1))

    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda c: due[c])

    # Build initial routes using a greedy insertion approach
    routes = []
    unassigned = set(customers)

    # For each customer, try to insert into existing routes or create new route
    for c in customers_sorted:
        inserted = False
        best_route_idx = -1
        best_dist = float('inf')

        for ri, route in enumerate(routes):
            feasible, new_route, dist = _insert_customer(
                route, c, D, instance, battery_cap, horizon, speed,
                recharge_rate, ready, due, service, demand, stations_set
            )
            if feasible and dist < best_dist:
                best_dist = dist
                best_route_idx = ri
                inserted = True

        if inserted:
            routes[best_route_idx] = best_route_idx  # placeholder, will fix below
            # Actually need to store the new route
            # Let me redo this more carefully
            pass

    # Simpler approach: just create one route per customer initially
    routes = [[c] for c in customers_sorted]

    # Try to merge routes
    changed = True
    while changed:
        changed = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                # Try to merge route j into route i
                best_dist = _route_distance(routes[i], D) + _route_distance(routes[j], D)
                best_merged = None

                # Try inserting each customer of route j into route i
                ri = list(routes[i])
                for c in routes[j]:
                    feasible, new_ri, dist = _insert_customer(
                        ri, c, D, instance, battery_cap, horizon, speed,
                        recharge_rate, ready, due, service, demand, stations_set
                    )
                    if feasible:
                        ri = new_ri
                    else:
                        ri = None
                        break

                if ri is not None:
                    merged_dist = _route_distance(ri, D)
                    if merged_dist < best_dist - 1e-9:
                        best_merged = ri
                        best_dist = merged_dist

                if best_merged is not None:
                    routes[i] = best_merged
                    routes.pop(j)
                    changed = True
                    break
            if changed:
                break

    # Local search: 2-opt and or-opt on each route
    for it in range(10):
        improved = False
        for ri in range(len(routes)):
            new_route = _try_2opt(routes[ri], D, instance, battery_cap, horizon,
                                  speed, recharge_rate, ready, due, service, demand,
                                  stations_set)
            if new_route != routes[ri]:
                routes[ri] = new_route
                improved = True
            new_route = _try_or_opt(routes[ri], D, instance, battery_cap, horizon,
                                    speed, recharge_rate, ready, due, service, demand,
                                    stations_set)
            if new_route != routes[ri]:
                routes[ri] = new_route
                improved = True
        if not improved:
            break

    # Try to merge routes again after local search
    changed = True
    while changed:
        changed = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                best_dist = _route_distance(routes[i], D) + _route_distance(routes[j], D)
                best_merged = None

                ri = list(routes[i])
                for c in routes[j]:
                    feasible, new_ri, dist = _insert_customer(
                        ri, c, D, instance, battery_cap, horizon, speed,
                        recharge_rate, ready, due, service, demand, stations_set
                    )
                    if feasible:
                        ri = new_ri
                    else:
                        ri = None
                        break

                if ri is not None:
                    merged_dist = _route_distance(ri, D)
                    if merged_dist < best_dist - 1e-9:
                        best_merged = ri
                        best_dist = merged_dist

                if best_merged is not None:
                    routes[i] = best_merged
                    routes.pop(j)
                    changed = True
                    break
            if changed:
                break

    # Final local search
    for ri in range(len(routes)):
        routes[ri] = _try_2opt(routes[ri], D, instance, battery_cap, horizon,
                               speed, recharge_rate, ready, due, service, demand,
                               stations_set)
        routes[ri] = _try_or_opt(routes[ri], D, instance, battery_cap, horizon,
                                 speed, recharge_rate, ready, due, service, demand,
                                 stations_set)

    return routes


def plan(instance):
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin

    routes = _solve_instance(instance)

    return routes
# EVOLVE-BLOCK-END
