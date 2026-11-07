"""EV routing with time windows – fresh design."""
import time as _time
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

# EVOLVE-BLOCK-START


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _simulate_route(nodes, coords, demands, ready, due, service,
                    stations, battery, speed, recharge_rate, horizon):
    """Simulate a route; return (total_dist, valid)."""
    n = len(coords)
    # node 0 is depot
    t = 0.0
    bat = float(battery)
    dist = 0.0
    cur = 0
    for v in nodes:
        d = _euclid(coords[cur], coords[v])
        if bat < d - 1e-9:
            return dist, False
        bat -= d
        t += d / speed
        dist += d
        cur = v
        if v in stations:
            # recharge
            charge_time = (bat) * recharge_rate
            t += charge_time
            bat = float(battery)
        else:
            # customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return dist, False
            t += service[v]
    # return to depot
    d = _euclid(coords[cur], coords[0])
    if bat < d - 1e-9:
        return dist, False
    bat -= d
    t += d / speed
    dist += d
    if t > horizon + 1e-9:
        return dist, False
    return dist, True


def _plan_instance(instance, deadline):
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_cust = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    n_nodes = len(coords)

    customers = list(range(1, n_cust + 1))

    # Precompute pairwise distances
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dist[i, j] = dist[j, i] = _euclid(coords[i], coords[j])

    # --- Step 1: Build initial routes via greedy nearest-neighbor with feasibility ---
    # Sort customers by due time
    cust_sorted = sorted(customers, key=lambda v: due[v])

    routes = []
    unvisited = set(customers)

    # We'll build routes by starting from depot, adding customers greedily
    # respecting capacity, battery, time windows
    while unvisited:
        route = []
        cur = 0
        t = 0.0
        bat = float(battery)
        load = 0.0
        # Try to add customers in order of due time
        for v in cust_sorted:
            if v not in unvisited:
                continue
            # Check if we can visit v next
            d = dist[cur][v]
            if bat < d - 1e-9:
                continue
            new_bat = bat - d
            new_t = t + d / speed
            if new_t < ready[v]:
                new_t = ready[v]
            if new_t > due[v] + 1e-9:
                continue
            new_t += service[v]
            new_load = load + demands[v]
            if new_load > capacity + 1e-9:
                continue
            # Check return to depot feasible
            d_ret = dist[v][0]
            if new_bat < d_ret - 1e-9:
                continue
            new_t_ret = new_t + d_ret / speed
            if new_t_ret > horizon + 1e-9:
                continue
            # Accept
            route.append(v)
            unvisited.discard(v)
            cur = v
            t = new_t
            bat = new_bat
            load = new_load
        if route:
            routes.append(route)
        else:
            # Fallback: pick the customer with earliest due
            v = min(unvisited, key=lambda x: due[x])
            routes.append([v])
            unvisited.discard(v)

    # --- Step 2: Insert charging stations where needed ---
    def _insert_stations(route, coords, dist, battery, speed, recharge_rate, horizon, ready, due, service, stations):
        """Given a route of customers, find a feasible sequence with stations."""
        if not route:
            return route, True

        # Try without stations first
        d, valid = _simulate_route(route, coords, demands, ready, due, service,
                                   stations, battery, speed, recharge_rate, horizon)
        if valid:
            return route, True

        # Try inserting stations at various positions
        # Build candidate station positions
        # We'll do a simple search: for each pair of consecutive nodes in the extended
        # sequence (depot, c1, c2, ..., depot), try inserting each station
        # This is expensive, so we limit to a few iterations

        # Strategy: simulate and when battery runs low, find nearest station
        # that allows reaching next customer or depot
        best_route = None
        best_dist = float('inf')

        # Try all permutations of station insertions is too expensive.
        # Instead, do a greedy repair: simulate, and when battery < needed,
        # detour to nearest station.

        def _greedy_repair(route):
            """Simulate and insert stations when battery is insufficient."""
            new_route = []
            cur = 0
            t = 0.0
            bat = float(battery)
            i = 0
            while i < len(route):
                v = route[i]
                d = dist[cur][v]
                if bat >= d - 1e-9:
                    bat -= d
                    t += d / speed
                    cur = v
                    new_route.append(v)
                    if v not in stations:
                        if t < ready[v]:
                            t = ready[v]
                        if t > due[v] + 1e-9:
                            return None, False
                        t += service[v]
                    else:
                        t += bat * recharge_rate
                        bat = float(battery)
                    i += 1
                else:
                    # Need a station
                    # Find nearest station from cur that we can reach
                    best_st = None
                    best_d = float('inf')
                    for s in stations:
                        ds = dist[cur][s]
                        if bat >= ds - 1e-9 and ds < best_d:
                            best_d = ds
                            best_st = s
                    if best_st is None:
                        return None, False
                    # Go to station
                    d_st = dist[cur][best_st]
                    bat -= d_st
                    t += d_st / speed
                    cur = best_st
                    t += bat * recharge_rate
                    bat = float(battery)
                    new_route.append(best_st)
                    # Don't increment i, try again
            # Return to depot
            d_ret = dist[cur][0]
            if bat < d_ret - 1e-9:
                return None, False
            bat -= d_ret
            t += d_ret / speed
            if t > horizon + 1e-9:
                return None, False
            total_d = 0.0
            for idx in range(len(new_route)):
                if idx == 0:
                    total_d += dist[0][new_route[0]]
                else:
                    total_d += dist[new_route[idx-1]][new_route[idx]]
            total_d += dist[new_route[-1]][0]
            return new_route, True

        result, ok = _greedy_repair(route)
        if ok:
            return result, True

        return route, False

    new_routes = []
    all_valid = True
    for r in routes:
        nr, ok = _insert_stations(r, coords, dist, battery, speed, recharge_rate, horizon,
                                  ready, due, service, stations)
        if not ok:
            all_valid = False
        new_routes.append(nr)

    if not all_valid:
        # Fallback to one customer per route
        return [[v] for v in customers]

    # --- Step 3: Local search to improve distance ---
    if _time.time() > deadline:
        return new_routes

    def _route_cost(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(1, len(route)):
            d += dist[route[i-1]][route[i]]
        d += dist[route[-1]][0]
        return d

    def _is_valid_route(route):
        if not route:
            return True
        # Check capacity
        load = sum(demands[v] for v in route if v not in stations)
        if load > capacity + 1e-9:
            return False
        d, valid = _simulate_route(route, coords, demands, ready, due, service,
                                   stations, battery, speed, recharge_rate, horizon)
        return valid

    total_cost = sum(_route_cost(r) for r in new_routes)
    improved = True
    max_iters = 50
    iters = 0
    while improved and iters < max_iters and _time.time() < deadline:
        improved = False
        iters += 1
        # Try 2-opt within routes
        for ri, route in enumerate(new_routes):
            if len(route) < 3:
                continue
            # 2-opt on customer-only positions (stations stay)
            # Simple: try reversing segments
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if _is_valid_route(new_route):
                        old_c = _route_cost(route)
                        new_c = _route_cost(new_route)
                        if new_c < old_c - 1e-9:
                            new_routes[ri] = new_route
                            total_cost += new_c - old_c
                            improved = True
        # Try moving a customer from one route to another
        for ri in range(len(new_routes)):
            if not new_routes[ri]:
                continue
            for ci, v in enumerate(new_routes[ri]):
                if v in stations:
                    continue
                for rj in range(len(new_routes)):
                    if rj == ri:
                        continue
                    # Try inserting v into route rj at each position
                    orig_route = new_routes[ri][:]
                    new_route_i = orig_route[:ci] + orig_route[ci+1:]
                    if not _is_valid_route(new_route_i):
                        continue
                    best_pos = None
                    best_c = float('inf')
                    for pos in range(len(new_routes[rj]) + 1):
                        cand = new_routes[rj][:pos] + [v] + new_routes[rj][pos:]
                        if _is_valid_route(cand):
                            c = _route_cost(cand) + _route_cost(new_route_i)
                            if c < best_c:
                                best_c = c
                                best_pos = pos
                    if best_pos is not None:
                        old_c = _route_cost(orig_route) + _route_cost(new_routes[rj])
                        new_c = _route_cost(new_route_i) + _route_cost(new_routes[rj][:best_pos] + [v] + new_routes[rj][best_pos:])
                        if new_c < old_c - 1e-9:
                            new_routes[ri] = new_route_i
                            new_routes[rj] = new_routes[rj][:best_pos] + [v] + new_routes[rj][best_pos:]
                            total_cost += new_c - old_c
                            improved = True

    return new_routes


def plan(instance):
    deadline = _time.time() + 110.0  # safety margin before 120s limit
    return _plan_instance(instance, deadline)


# EVOLVE-BLOCK-END
