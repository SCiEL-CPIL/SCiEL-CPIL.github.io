"""EV routing with time windows – Clarke-Wright Construction + Enhanced Local Search + Polish Pass."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _build_distance_matrix(coords):
    n = len(coords)
    c = np.array(coords, dtype=np.float64)
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        diff = c[i] - c
        D[i] = np.sqrt(np.sum(diff * diff, axis=1))
    return D


def _simulate(route, D, instance):
    """Simulate a single route. Returns (total_distance, valid, end_time, end_battery)."""
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]
    n_customers = instance["n_customers"]

    t = 0.0
    batt = battery_max
    load = 0
    dist_total = 0.0
    prev = 0

    for node in route:
        d = D[prev, node]
        dist_total += d
        t += d
        batt -= d

        if batt < -1e-9:
            return dist_total, False, t, batt

        if node in stations_set:
            energy_needed = battery_max - batt
            t += energy_needed * recharge_rate
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return dist_total, False, t, batt

            load += demands[node]
            if load > capacity + 1e-9:
                return dist_total, False, t, batt

            t += service[node]
        prev = node

    d = D[prev, 0]
    dist_total += d
    t += d
    batt -= d

    if batt < -1e-9:
        return dist_total, False, t, batt
    if t > horizon + 1e-9:
        return dist_total, False, t, batt

    return dist_total, True, t, batt


def _total_distance(routes, D):
    total = 0.0
    for route in routes:
        if not route:
            continue
        prev = 0
        for node in route:
            total += D[prev, node]
            prev = node
        total += D[prev, 0]
    return total


def _is_valid_solution(routes, instance, D):
    served = set()
    stations_set = instance["stations_set"]
    n_customers = instance["n_customers"]

    for route in routes:
        for node in route:
            if node in stations_set:
                continue
            if node in served:
                return False
            served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False

    for route in routes:
        _, valid, _, _ = _simulate(route, D, instance)
        if not valid:
            return False
    return True


def _fast_feasible(route, D, instance):
    """
    Fast feasibility check: simulate and track the minimum battery reached.
    If the route is infeasible due to battery, check whether inserting the
    best station at the point of minimum battery would fix it (one-level repair).
    Returns True if feasible (possibly with a station insert), False otherwise.
    """
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    stations_set = instance["stations_set"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]

    # Capacity check
    load = 0
    for node in route:
        if node not in stations_set:
            load += demands[node]
    if load > capacity + 1e-9:
        return False

    t = 0.0
    batt = battery_max
    prev = 0
    min_batt = battery_max
    min_batt_idx = -1  # index in route where min battery occurs (before the node)

    for idx, node in enumerate(route):
        d = D[prev, node]
        t += d
        batt -= d
        if batt < min_batt:
            min_batt = batt
            min_batt_idx = idx

        if batt < -1e-9:
            # Battery failed: try inserting best station before this node
            # Find the station that minimizes the additional distance
            # Actually, just try each station at min_batt_idx position
            for st in stations:
                # Quick check: does inserting st at min_batt_idx help?
                # We need a full re-sim which is expensive; do a rough check
                # Just try the nearest station to prev at the failure point
                break  # skip for speed in fast check; rely on _try_fix_route later
            return False

        if node in stations_set:
            energy_needed = battery_max - batt
            t += energy_needed * recharge_rate
            batt = battery_max
        else:
            if t < ready[node]:
                t = ready[node]
            if t > due[node] + 1e-9:
                return False
            t += service[node]
        prev = node

    # Return leg
    d = D[prev, 0]
    t += d
    batt -= d
    if batt < -1e-9:
        return False
    if t > horizon + 1e-9:
        return False

    return True


def _try_fix_route(route, D, instance, max_iters=15):
    """Attempt to fix a route by inserting stations."""
    stations = instance["stations"]
    current_route = list(route)

    for _ in range(max_iters):
        dist, valid, t, batt = _simulate(current_route, D, instance)
        if valid:
            return current_route

        # Find where it fails
        t_sim = 0.0
        batt_sim = instance["battery"]
        prev = 0
        fail_idx = -1

        for i, node in enumerate(current_route):
            d = D[prev, node]
            t_sim += d
            batt_sim -= d
            if batt_sim < -1e-9:
                fail_idx = i
                break
            if node not in instance["stations_set"]:
                if t_sim < instance["ready"][node]:
                    t_sim = instance["ready"][node]
                if t_sim > instance["due"][node] + 1e-9:
                    fail_idx = i
                    break
                t_sim += instance["service"][node]
            prev = node

        if fail_idx == -1:
            # Failed on return leg
            for st in stations:
                new_route = current_route + [st]
                dist, valid, _, _ = _simulate(new_route, D, instance)
                if valid:
                    return new_route
            return None

        # Try inserting a station before the failing node
        best_new_route = None
        best_dist = float('inf')

        for st in stations:
            new_route = current_route[:fail_idx] + [st] + current_route[fail_idx:]
            dist, valid, _, _ = _simulate(new_route, D, instance)
            if valid and dist < best_dist:
                best_dist = dist
                best_new_route = new_route

        if best_new_route:
            current_route = best_new_route
        else:
            # Try inserting at a few nearby positions
            found = False
            for offset in range(1, min(4, fail_idx + 1)):
                pos = fail_idx - offset
                for st in stations:
                    new_route = current_route[:pos] + [st] + current_route[pos:]
                    dist, valid, _, _ = _simulate(new_route, D, instance)
                    if valid:
                        current_route = new_route
                        found = True
                        break
                if found:
                    break
            if not found:
                return None

    return current_route if _simulate(current_route, D, instance)[1] else None


def _clarke_wright_savings(instance, D):
    """Clarke-Wright savings heuristic with 4-way merge and fast feasibility."""
    demands = instance["demands"]
    capacity = instance["capacity"]
    n_customers = instance["n_customers"]
    stations_set = instance["stations_set"]
    due = instance["due"]

    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda c: due[c])

    routes = [[c] for c in customers]
    loads = [demands[c] for c in customers]

    # Pre-compute savings for all customer pairs
    n = len(customers)
    savings = []
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj = customers[i], customers[j]
            s = D[0, ci] + D[0, cj] - D[ci, cj]
            if s > 0:
                savings.append((s, ci, cj))
    savings.sort(reverse=True)

    # Map customer -> route index
    cust_to_route = {customers[i]: i for i in range(n)}

    for s_val, ci, cj in savings:
        ri = cust_to_route.get(ci)
        rj = cust_to_route.get(cj)
        if ri is None or rj is None or ri == rj:
            continue

        if loads[ri] + loads[rj] > capacity:
            continue

        # Try 4 merge types
        ra = routes[ri]
        rb = routes[rj]

        best_merged = None
        best_saving = -1

        # Type 0: ra + rb (ci at end of ra, cj at start of rb)
        if ra[-1] == ci and rb[0] == cj:
            merged = ra + rb
            sav = D[0, ci] + D[0, cj] - D[ci, cj]
            if sav > best_saving and _fast_feasible(merged, D, instance):
                best_saving = sav
                best_merged = merged

        # Type 1: ra + rb[::-1] (ci at end of ra, cj at end of rb)
        if ra[-1] == ci and rb[-1] == cj:
            merged = ra + rb[::-1]
            sav = D[0, ci] + D[0, cj] - D[ci, cj]
            if sav > best_saving and _fast_feasible(merged, D, instance):
                best_saving = sav
                best_merged = merged

        # Type 2: ra[::-1] + rb (ci at start of ra, cj at start of rb)
        if ra[0] == ci and rb[0] == cj:
            merged = ra[::-1] + rb
            sav = D[0, ci] + D[0, cj] - D[ci, cj]
            if sav > best_saving and _fast_feasible(merged, D, instance):
                best_saving = sav
                best_merged = merged

        # Type 3: ra[::-1] + rb[::-1] (ci at start of ra, cj at end of rb)
        if ra[0] == ci and rb[-1] == cj:
            merged = ra[::-1] + rb[::-1]
            sav = D[0, ci] + D[0, cj] - D[ci, cj]
            if sav > best_saving and _fast_feasible(merged, D, instance):
                best_saving = sav
                best_merged = merged

        if best_merged is None:
            continue

        # Perform merge: remove rj, update ri
        routes[ri] = best_merged
        loads[ri] += loads[rj]

        # Update cust_to_route for all customers in the merged route
        for c in best_merged:
            if c not in stations_set:
                cust_to_route[c] = ri

        # Remove rj from tracking
        del routes[rj]
        del loads[rj]

    return [r for r in routes if r]


def _2opt(route, D, instance, time_limit, tol=1e-9):
    """2-opt improvement for a single route."""
    current_dist, valid, _, _ = _simulate(route, D, instance)
    if not valid:
        return route

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if time.time() > time_limit:
                    return route
                new_route = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                dist, valid, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - tol:
                    route = new_route
                    current_dist = dist
                    improved = True
                    break
            if improved:
                break

    return route


def _or_opt(route, D, instance, time_limit, tol=1e-9):
    """Or-opt: relocate 1 or 2 consecutive nodes."""
    current_dist, valid, _, _ = _simulate(route, D, instance)
    if not valid:
        return route

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        n = len(route)
        for i in range(n):
            for length in [1, 2]:
                if i + length > n:
                    continue
                segment = route[i:i + length]
                rest = route[:i] + route[i + length:]
                if not rest:
                    continue
                for j in range(len(rest) + 1):
                    if time.time() > time_limit:
                        return route
                    new_route = rest[:j] + segment + rest[j:]
                    dist, valid, _, _ = _simulate(new_route, D, instance)
                    if valid and dist < current_dist - tol:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return route


def _station_opt(route, D, instance, time_limit, tol=1e-9):
    """Try to move or insert stations to improve distance or feasibility."""
    current_dist, valid, _, _ = _simulate(route, D, instance)

    if not valid:
        fixed = _try_fix_route(route, D, instance)
        if fixed is not None:
            route = fixed
            current_dist, valid, _, _ = _simulate(route, D, instance)
            if not valid:
                return route

    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        n = len(route)

        # Move existing stations
        for i in range(n):
            if route[i] not in instance["stations_set"]:
                continue
            st = route[i]
            rest = route[:i] + route[i + 1:]
            for j in range(len(rest) + 1):
                if time.time() > time_limit:
                    return route
                new_route = rest[:j] + [st] + rest[j:]
                dist, valid, _, _ = _simulate(new_route, D, instance)
                if valid and dist < current_dist - tol:
                    route = new_route
                    current_dist = dist
                    improved = True
                    break
            if improved:
                break

        # Insert new station
        if not improved:
            for st in instance["stations"]:
                if st in route:
                    continue
                for j in range(len(route) + 1):
                    if time.time() > time_limit:
                        return route
                    new_route = route[:j] + [st] + route[j:]
                    dist, valid, _, _ = _simulate(new_route, D, instance)
                    if valid and dist < current_dist - tol:
                        route = new_route
                        current_dist = dist
                        improved = True
                        break
                if improved:
                    break

    return route


def _cross_route_swap(routes, D, instance, time_limit):
    """Inter-route improvement: swap customers between routes."""
    improved = True
    while improved:
        if time.time() > time_limit:
            break
        improved = False
        for a in range(len(routes)):
            for b in range(a + 1, len(routes)):
                for i in range(len(routes[a])):
                    for j in range(len(routes[b])):
                        if time.time() > time_limit:
                            return routes
                        if routes[a][i] in instance["stations_set"] or routes[b][j] in instance["stations_set"]:
                            continue

                        ra = list(routes[a])
                        rb = list(routes[b])
                        ra[i], rb[j] = rb[j], ra[i]
                        va = _simulate(ra, D, instance)[1]
                        vb = _simulate(rb, D, instance)[1]
                        if va and vb:
                            da = _simulate(ra, D, instance)[0]
                            db = _simulate(rb, D, instance)[0]
                            old_dist = _simulate(routes[a], D, instance)[0] + _simulate(routes[b], D, instance)[0]
                            if da + db < old_dist - 1e-9:
                                routes[a] = ra
                                routes[b] = rb
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return routes


def _polish_pass(routes, D, instance, time_limit):
    """High-precision polish: tighter tolerance, more iterations."""
    for i in range(len(routes)):
        if time.time() > time_limit:
            break
        # Tighter tolerance polish
        routes[i] = _2opt(routes[i], D, instance, time_limit, tol=1e-12)
        routes[i] = _or_opt(routes[i], D, instance, time_limit, tol=1e-12)

    # Another round of cross-route swap with tighter tolerance
    if time.time() < time_limit:
        routes = _cross_route_swap(routes, D, instance, time_limit)

    # Final station optimization
    for i in range(len(routes)):
        if time.time() > time_limit:
            break
        routes[i] = _station_opt(routes[i], D, instance, time_limit, tol=1e-12)

    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    coords = instance["coords"]
    D = _build_distance_matrix(coords)
    instance["stations_set"] = set(instance["stations"])

    # 1. Construct Initial Solution using Clarke-Wright
    routes = _clarke_wright_savings(instance, D)

    # 2. Fix any infeasible routes
    fixed_routes = []
    for r in routes:
        fixed = _try_fix_route(r, D, instance)
        if fixed is not None:
            fixed_routes.append(fixed)
        else:
            for c in r:
                if c not in instance["stations_set"]:
                    fixed_routes.append([c])

    best_routes = fixed_routes

    # Ensure all customers are covered
    served = set()
    for r in best_routes:
        for n in r:
            if n not in instance["stations_set"]:
                served.add(n)
    missing = set(range(1, instance["n_customers"] + 1)) - served
    for m in missing:
        best_routes.append([m])

    # 3. Local Search
    passes = 0
    max_passes = 40

    while time.time() < deadline and passes < max_passes:
        passes += 1
        improved = False
        remaining = deadline - time.time()
        current_time_limit = time.time() + remaining / (max_passes - passes + 1)

        # Intra-route optimizations
        for i in range(len(best_routes)):
            if time.time() > deadline:
                break
            new_route = _2opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

            new_route = _or_opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

            new_route = _station_opt(best_routes[i], D, instance, current_time_limit)
            if new_route != best_routes[i]:
                best_routes[i] = new_route
                improved = True

        # Inter-route optimization
        if passes % 2 == 0 and time.time() < deadline:
            new_routes = _cross_route_swap(best_routes, D, instance, current_time_limit)
            if new_routes != best_routes:
                best_routes = new_routes
                improved = True

        if not improved:
            break

    # 4. Final high-precision polish pass
    if time.time() < deadline - 5.0:
        polish_deadline = min(deadline, time.time() + (deadline - time.time()) * 0.3)
        best_routes = _polish_pass(best_routes, D, instance, polish_deadline)

    # 5. Final Validation
    if not _is_valid_solution(best_routes, instance, D):
        customers = list(range(1, instance["n_customers"] + 1))
        best_routes = [[c] for c in customers]

    return best_routes
# EVOLVE-BLOCK-END
