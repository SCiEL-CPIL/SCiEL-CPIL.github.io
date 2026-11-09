"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random


def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            D[i, j] = d
            D[j, i] = d
    return D


def _route_time_and_battery(route, D, instance):
    """Return (total_time, total_dist, min_battery, feasible) for a route."""
    speed = instance.get("speed", 1)
    battery_cap = instance.get("battery", 300)
    recharge_rate = instance.get("recharge_rate", 0.5)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    t = 0.0
    bat = battery_cap
    dist = 0.0
    prev = 0
    min_bat = battery_cap
    for v in route:
        d = D[prev, v]
        t += d / speed
        bat -= d
        dist += d
        if bat < -1e-9:
            return t, dist, bat, False
        min_bat = min(min_bat, bat)
        if v in stations:
            energy_needed = battery_cap - bat
            t += energy_needed * recharge_rate
            bat = battery_cap
        else:
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return t, dist, bat, False
            t += service[v]
        prev = v
    d = D[prev, 0]
    t += d / speed
    bat -= d
    dist += d
    if bat < -1e-9:
        return t, dist, bat, False
    min_bat = min(min_bat, bat)
    if t > horizon + 1e-9:
        return t, dist, bat, False
    return t, dist, min_bat, True


def _route_feasible(route, D, instance):
    if not route:
        return True
    cap = instance.get("capacity", 100)
    demands = instance["demands"]
    if sum(demands[v] for v in route) > cap + 1e-9:
        return False
    _, _, _, feas = _route_time_and_battery(route, D, instance)
    return feas


def _route_dist(route, D):
    if not route:
        return 0.0
    total = D[0, route[0]]
    for i in range(len(route) - 1):
        total += D[route[i], route[i + 1]]
    total += D[route[-1], 0]
    return total


def _total_dist(routes, D):
    return sum(_route_dist(r, D) for r in routes)


def _savings_construct(instance, D, deadline):
    """Savings-based construction: start with individual routes, merge by savings."""
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    battery_cap = instance.get("battery", 300)
    speed = instance.get("speed", 1)
    recharge_rate = instance.get("recharge_rate", 0.5)
    cap = instance.get("capacity", 100)
    horizon = instance.get("horizon", 1000)
    stations = set(instance.get("stations", []))
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]

    # Start with each customer as its own route
    routes = [[v] for v in customers]
    route_of = {}
    for i, r in enumerate(routes):
        for v in r:
            route_of[v] = i

    # Compute savings: saving of merging i and j = D[0,i] + D[0,j] - D[i,j]
    # But we need to check feasibility.
    # Sort customers by due time for tie-breaking
    customers_sorted = sorted(customers, key=lambda v: due[v])

    # Compute all pairwise savings
    savings_list = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            saving = D[0, ci] + D[0, cj] - D[ci, cj]
            if saving > 0:
                savings_list.append((saving, i, j))

    savings_list.sort(key=lambda x: -x[0])

    # Try to merge routes in order of savings
    for saving, ci, cj in savings_list:
        if time.time() > deadline:
            break
        ri = route_of.get(ci)
        rj = route_of.get(cj)
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Try all combinations of merging route_i and route_j
        # The merged route can be:
        # [route_i + route_j], [route_j + route_i],
        # [reversed(route_i) + route_j], [route_i + reversed(route_j)], etc.
        # We try to find a feasible merged route that minimizes distance.

        best_merged = None
        best_dist = float('inf')

        # Generate candidate merged routes
        candidates = []
        # Append route_j to end of route_i
        candidates.append(route_i + route_j)
        # Append route_i to end of route_j
        candidates.append(route_j + route_i)
        # Reverse route_i and append route_j
        candidates.append(route_i[::-1] + route_j)
        # Reverse route_j and append route_i
        candidates.append(route_j[::-1] + route_i)

        for cand in candidates:
            if not _route_feasible(cand, D, instance):
                continue
            d = _route_dist(cand, D)
            if d < best_dist:
                best_dist = d
                best_merged = cand

        if best_merged is not None:
            # Merge: replace routes[ri] with best_merged, remove routes[rj]
            routes[ri] = best_merged
            routes[rj] = []
            for v in best_merged:
                route_of[v] = ri

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _2opt(route, D, instance, max_iter=300):
    if len(route) < 2:
        return route
    best = route[:]

    for _ in range(max_iter):
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                a = best[i - 1] if i > 0 else 0
                b = best[i]
                c = best[j]
                d = best[j + 1] if j + 1 < len(best) else 0
                old_seg = D[a, b] + D[c, d]
                new_seg = D[a, c] + D[b, d]
                if new_seg >= old_seg - 1e-9:
                    continue
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                if not _route_feasible(new_route, D, instance):
                    continue
                best = new_route
                improved = True
                break
            if improved:
                break
        if not improved:
            break
    return best


def _or_opt(route, D, instance, max_iter=150):
    if len(route) < 3:
        return route
    best = route[:]
    best_dist = _route_dist(best, D)

    for _ in range(max_iter):
        improved = False
        for i in range(len(best)):
            node = best[i]
            for j in range(len(best)):
                if j == i or j == i - 1:
                    continue
                new_route = best[:i] + best[i + 1:]
                ins_j = j if i > j else j - 1
                if ins_j < 0:
                    ins_j = 0
                if ins_j > len(new_route):
                    ins_j = len(new_route)
                candidate = new_route[:ins_j] + [node] + new_route[ins_j:]
                if candidate == best:
                    continue
                if not _route_feasible(candidate, D, instance):
                    continue
                cand_dist = _route_dist(candidate, D)
                if cand_dist < best_dist - 1e-9:
                    best = candidate
                    best_dist = cand_dist
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def _relocate_inter(routes, D, instance, max_iters=50):
    """Move a customer from one route to another."""
    for _ in range(max_iters):
        if time.time() > 150 - 10:
            break
        improved = False
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                for ai in range(len(routes[i])):
                    v = routes[i][ai]
                    new_i = routes[i][:ai] + routes[i][ai+1:]
                    for aj in range(len(routes[j]) + 1):
                        new_j = routes[j][:aj] + [v] + routes[j][aj:]
                        if not new_i and not new_j:
                            continue
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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
    return routes


def _swap_inter(routes, D, instance, max_iters=30):
    """Swap two customers between routes."""
    for _ in range(max_iters):
        if time.time() > 150 - 10:
            break
        improved = False
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                for ai in range(len(routes[i])):
                    for aj in range(len(routes[j])):
                        vi = routes[i][ai]
                        vj = routes[j][aj]
                        new_i = routes[i][:]
                        new_j = routes[j][:]
                        new_i[ai] = vj
                        new_j[aj] = vi
                        if not _route_feasible(new_i, D, instance):
                            continue
                        if not _route_feasible(new_j, D, instance):
                            continue
                        old_d = _route_dist(routes[i], D) + _route_dist(routes[j], D)
                        new_d = _route_dist(new_i, D) + _route_dist(new_j, D)
                        if new_d < old_d - 1e-9:
                            routes[i] = new_i
                            routes[j] = new_j
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
    return routes


def _lns_destroy_repair(routes, D, instance, rng, num_removals=3):
    """LNS: remove random customers, reinsert greedily."""
    best_routes = [r[:] for r in routes]
    best_dist = _total_dist(best_routes, D)

    # Remove customers
    all_customers = [v for r in best_routes for v in r]
    if not all_customers:
        return best_routes

    num_rem = min(num_removals, len(all_customers))
    removed = rng.sample(all_customers, num_rem)
    removed_set = set(removed)

    # Build routes without removed customers
    work_routes = []
    for r in best_routes:
        new_r = [v for v in r if v not in removed_set]
        if new_r:
            work_routes.append(new_r)

    # Reinsert removed customers greedily
    for v in removed:
        best_pos = None
        best_cost = float('inf')
        for ri, r in enumerate(work_routes):
            for pos in range(len(r) + 1):
                candidate = r[:pos] + [v] + r[pos:]
                if not _route_feasible(candidate, D, instance):
                    continue
                cost = _route_dist(candidate, D) - _route_dist(r, D)
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (ri, pos)

        if best_pos is not None:
            ri, pos = best_pos
            work_routes[ri] = work_routes[ri][:pos] + [v] + work_routes[ri][pos:]
        else:
            # Put in a new route
            work_routes.append([v])

    # Check if better
    new_dist = _total_dist(work_routes, D)
    if new_dist < best_dist - 1e-9:
        best_routes = work_routes
        best_dist = new_dist

    return best_routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140

    coords = instance["coords"]
    D = _dist_matrix(coords)

    # Use savings-based construction
    routes = _savings_construct(instance, D, deadline)

    if time.time() > deadline:
        return routes

    # Local search: 2-opt and OR-opt on each route
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, instance)
        routes[r_idx] = _or_opt(routes[r_idx], D, instance)

    if time.time() > deadline:
        return routes

    # Inter-route moves
    routes = _relocate_inter(routes, D, instance)
    routes = _swap_inter(routes, D, instance)

    if time.time() > deadline:
        return routes

    # LNS with random restarts
    rng = random.Random(42)
    for iteration in range(50):
        if time.time() > deadline:
            break
        routes = _lns_destroy_repair(routes, D, instance, rng, num_removals=3)
        # Quick local search after LNS
        for r_idx in range(len(routes)):
            if time.time() > deadline:
                break
            routes[r_idx] = _2opt(routes[r_idx], D, instance, max_iter=100)

    if time.time() > deadline:
        return routes

    # Final polish
    routes = _relocate_inter(routes, D, instance)
    routes = _swap_inter(routes, D, instance)

    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _2opt(routes[r_idx], D, instance)
        routes[r_idx] = _or_opt(routes[r_idx], D, instance)

    return routes
# EVOLVE-BLOCK-END
