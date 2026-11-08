"""EV-routing solver with construction + local search."""
import math
import time

# EVOLVE-BLOCK-START


def _euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _check_route(route, instance):
    """Return (valid, total_distance) for a single route (list of customer/station ids, depot implicit)."""
    coords = instance["coords"]
    demands = instance["demand"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    total_dist = 0.0
    total_demand = 0.0
    battery = battery_cap
    t = 0.0

    # depot at index 0
    prev = 0
    for node in route:
        d = _euclid(coords[prev], coords[node])
        total_dist += d
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return False, total_dist
        if node in stations:
            # recharge
            energy_needed = battery_cap - battery
            t += energy_needed * recharge_rate
            battery = battery_cap
        else:
            # customer
            total_demand += demands[node]
            if total_demand > capacity + 1e-9:
                return False, total_dist
            t = max(t, ready[node])
            if t > due[node] + 1e-9:
                return False, total_dist
            t += service[node]
        prev = node

    # return to depot
    d = _euclid(coords[prev], coords[0])
    total_dist += d
    battery -= d
    t += d / speed
    if battery < -1e-9:
        return False, total_dist
    if t > horizon + 1e-9:
        return False, total_dist

    return True, total_dist


def _check_all(routes, instance):
    served = set()
    for route in routes:
        valid, dist = _check_route(route, instance)
        if not valid:
            return False
        for node in route:
            if node not in instance["stations"]:
                served.add(node)
    customers = set(range(1, instance["n_customers"] + 1))
    return served == customers


def _construct_swept(instance):
    """Swept-clustering: sort by angle from depot, cluster into groups, build routes."""
    coords = instance["coords"]
    n_cust = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    demands = instance["demand"]

    # Sort customers by angle from depot
    depot = coords[0]
    def angle(c):
        return math.atan2(coords[c][1] - depot[1], coords[c][0] - depot[0])
    cust_sorted = list(range(1, n_cust + 1))
    cust_sorted.sort(key=angle)

    # Greedy: add customers in angular order to current route if capacity allows
    routes = []
    cur_route = []
    cur_demand = 0.0
    for c in cust_sorted:
        if cur_demand + demands[c] <= capacity:
            cur_route.append(c)
            cur_demand += demands[c]
        else:
            if cur_route:
                routes.append(cur_route)
            cur_route = [c]
            cur_demand = demands[c]
    if cur_route:
        routes.append(cur_route)

    # Insert charging stations if needed for feasibility
    routes = _ensure_battery_feasible(routes, instance)
    return routes


def _ensure_battery_feasible(routes, instance):
    """For each route, if battery infeasible, insert nearest station at worst point. Repeat."""
    battery_cap = instance["battery"]
    coords = instance["coords"]
    stations = instance["stations"]
    max_iter = 50
    for ri, route in enumerate(routes):
        for _ in range(max_iter):
            valid, _ = _check_route(route, instance)
            if valid:
                break
            # Find where battery goes lowest; insert nearest station there
            battery = battery_cap
            min_bat = float('inf')
            min_idx = 0
            prev = 0
            for i, node in enumerate(route):
                d = _euclid(coords[prev], coords[node])
                battery -= d
                if battery < min_bat:
                    min_bat = battery
                    min_idx = i
                prev = node
            # pick nearest station to coords[route[min_idx]]
            best_st = min(stations, key=lambda s: _euclid(coords[s], coords[route[min_idx]]))
            route.insert(min_idx + 1, best_st)
    return routes


def _local_search(routes, instance, time_limit=140.0, start_time=None):
    """2-opt within routes, or-opt, inter-route moves."""
    if start_time is None:
        start_time = time.time()
    coords = instance["coords"]
    stations = set(instance["stations"])
    n_cust = instance["n_customers"]
    all_customers = list(range(1, n_cust + 1))

    def route_dist(route):
        total = 0.0
        prev = 0
        for node in route:
            total += _euclid(coords[prev], coords[node])
            prev = node
        total += _euclid(coords[prev], coords[0])
        return total

    def is_valid_route(route):
        valid, _ = _check_route(route, instance)
        return valid

    def try_2opt(route):
        """Standard 2-opt on customer subsequence (stations stay relative? Actually stations can move too)."""
        # We'll do 2-opt on the full sequence including stations
        n = len(route)
        if n < 2:
            return False
        improved = False
        for i in range(n - 1):
            if time.time() - start_time > time_limit:
                return improved
            for j in range(i + 1, n):
                # reverse route[i..j]
                a = route[i - 1] if i > 0 else 0
                b = route[i]
                c = route[j]
                d = route[j + 1] if j + 1 < n else 0
                delta = _euclid(coords[a], coords[d]) + _euclid(coords[b], coords[c]) - \
                         _euclid(coords[a], coords[b]) - _euclid(coords[c], coords[d])
                if delta < -1e-9:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    if is_valid_route(route):
                        improved = True
                    else:
                        route[i:j + 1] = route[i:j + 1][::-1]  # revert
        return improved

    def try_or_opt(route, other_routes, ri):
        """Move a segment of length 1-3 to another position in same or other route."""
        improved = False
        n = len(route)
        if n == 0:
            return improved
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                if time.time() - start_time > time_limit:
                    return improved
                seg = route[i:i + seg_len]
                if any(s in stations for s in seg):
                    continue  # skip segments with stations for simplicity
                # remove
                rest = route[:i] + route[i + seg_len:]
                # try inserting into same route at all positions
                for ins in range(len(rest) + 1):
                    if ins <= i or ins >= i + seg_len:
                        continue
                    cand = rest[:ins] + seg + rest[ins:]
                    if is_valid_route(cand):
                        if route_dist(cand) < route_dist(route) - 1e-9:
                            route[:] = cand
                            improved = True
                            break
                if improved:
                    break
                # try inserting into other routes
                for rj in range(len(other_routes)):
                    if rj == ri:
                        continue
                    target = other_routes[rj]
                    # try all insertion positions
                    best_pos = -1
                    best_d = float('inf')
                    for ins in range(len(target) + 1):
                        cand = target[:ins] + seg + target[ins:]
                        if is_valid_route(cand):
                            d = route_dist(cand) + route_dist(rest)
                            if d < best_d:
                                best_d = d
                                best_pos = ins
                    if best_pos >= 0 and best_d < route_dist(route) + route_dist(target) - 1e-9:
                        other_routes[rj] = target[:best_pos] + seg + target[best_pos:]
                        route[:] = rest
                        if not rest:
                            # remove empty route
                            pass
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        return improved

    def try_inter_swap(routes, ri, rj):
        """Swap two customers between routes ri and rj."""
        a = routes[ri]
        b = routes[rj]
        improved = False
        for i in range(len(a)):
            if a[i] in stations:
                continue
            for j in range(len(b)):
                if b[j] in stations:
                    continue
                a[i], b[j] = b[j], a[i]
                if is_valid_route(a) and is_valid_route(b):
                    if route_dist(a) + route_dist(b) < _saved_dist[ri] + _saved_dist[rj] - 1e-9:
                        improved = True
                    else:
                        a[i], b[j] = b[j], a[i]
                else:
                    a[i], b[j] = b[j], a[i]
                if improved:
                    return True
        return improved

    # Main loop
    no_improve = 0
    while time.time() - start_time < time_limit:
        if no_improve > 5:
            break
        improved = False
        # Compute current distances
        _saved_dist = [route_dist(r) for r in routes]

        # 2-opt
        for ri in range(len(routes)):
            if time.time() - start_time > time_limit:
                return routes
            if try_2opt(routes[ri]):
                improved = True

        # or-opt
        for ri in range(len(routes)):
            if time.time() - start_time > time_limit:
                return routes
            if try_or_opt(routes[ri], routes, ri):
                improved = True

        # Remove empty routes
        routes[:] = [r for r in routes if r]

        # Inter-route swaps (limited)
        if len(routes) > 1:
            for ri in range(len(routes)):
                if time.time() - start_time > time_limit:
                    return routes
                for rj in range(ri + 1, len(routes)):
                    if try_inter_swap(routes, ri, rj):
                        improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1

    return routes


def plan(instance):
    t0 = time.time()
    deadline = 140.0  # safety margin within 150s

    # Build initial solution
    routes = _construct_swept(instance)

    # Local search with time budget
    remaining = deadline - (time.time() - t0)
    if remaining > 5:
        routes = _local_search(routes, instance, time_limit=remaining, start_time=t0)

    # Final validation; if invalid, fall back to single-customer routes
    if not _check_all(routes, instance):
        customers = list(range(1, instance["n_customers"] + 1))
        customers.sort(key=lambda v: instance["due"][v])
        routes = [[v] for v in customers]

    return routes
# EVOLVE-BLOCK-END
