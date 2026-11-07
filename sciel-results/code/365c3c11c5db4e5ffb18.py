"""EV routing with time windows and battery constraints - refined approach."""
import numpy as np
import time
import random
import math

DEADLINE_SECONDS = 140.0
EPS = 1e-9


def _build_dist_matrix(coords):
    coords = np.array(coords, dtype=np.float64)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _simulate_route(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]

    total_demand = 0
    for node in route:
        if 1 <= node <= n_customers:
            total_demand += demands[node]
    if total_demand > capacity:
        return False, 0.0, 0.0, 0.0

    battery = battery_max
    time_at = 0.0
    distance = 0.0
    current_node = 0

    for node in route:
        d = dist_matrix[current_node, node]
        time_at += d / speed
        battery -= d
        distance += d

        if battery < -EPS:
            return False, 0.0, 0.0, 0.0

        if 1 <= node <= n_customers:
            if time_at < ready[node]:
                time_at = ready[node]
            if time_at > due[node] + EPS:
                return False, 0.0, 0.0, 0.0
            time_at += service[node]
        elif node in stations_set:
            energy_needed = battery_max - battery
            if energy_needed > 0:
                time_at += energy_needed * recharge_rate
            battery = battery_max

        current_node = node

    d = dist_matrix[current_node, 0]
    time_at += d / speed
    battery -= d
    distance += d

    if battery < -EPS:
        return False, 0.0, 0.0, 0.0
    if time_at > horizon + EPS:
        return False, 0.0, 0.0, 0.0

    return True, distance, time_at, battery


def _validate_solution(routes, instance):
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    dist_matrix = _build_dist_matrix(coords)
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    served = set()
    total_distance = 0.0

    for route in routes:
        valid, dist, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                             stations_set, battery_max, speed,
                                             recharge_rate, horizon, capacity)
        if not valid:
            return False, 0.0
        total_distance += dist
        for node in route:
            if 1 <= node <= n_customers:
                if node in served:
                    return False, 0.0
                served.add(node)

    if served != set(range(1, n_customers + 1)):
        return False, 0.0

    return True, total_distance


def _compute_total_distance(routes, dist_matrix):
    total = 0.0
    for route in routes:
        current = 0
        for node in route:
            total += dist_matrix[current, node]
            current = node
        total += dist_matrix[current, 0]
    return total


def _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                    battery_max, speed, recharge_rate, horizon, capacity):
    valid, _, _, _ = _simulate_route(route, dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed,
                                      recharge_rate, horizon, capacity)
    return valid


def _savings_construct(instance, dist_matrix, n_customers, stations_set,
                        battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    """Clarke-Wright savings construction."""
    random.seed(seed)
    demands = instance["demands"]

    # Start with individual routes
    routes = [[i] for i in range(1, n_customers + 1)]
    loads = [demands[i] for i in range(1, n_customers + 1)]

    # Compute savings
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci, cj = i + 1, j + 1
            s = dist_matrix[0, ci] + dist_matrix[0, cj] - dist_matrix[ci, cj]
            savings.append((s, ci, cj))
    savings.sort(reverse=True)

    # Union-find for tracking which route each customer is in
    route_of = {i: i for i in range(1, n_customers + 1)}
    parent = {i: i for i in range(1, n_customers + 1)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for s, ci, cj in savings:
        if s <= 0:
            break
        ri, rj = find(ci), find(cj)
        if ri == rj:
            continue
        # Find the routes
        route_i = None
        route_j = None
        for idx, r in enumerate(routes):
            if r and ri in r:
                route_i = idx
                break
        for idx, r in enumerate(routes):
            if r and rj in r:
                route_j = idx
                break
        if route_i is None or route_j is None or route_i == route_j:
            continue

        # Check capacity
        if loads[route_i] + loads[route_j] > capacity:
            continue

        # Try to merge: ci must be at end of route_i, cj at start of route_j (or vice versa)
        merged = None
        r_i = routes[route_i]
        r_j = routes[route_j]

        # Option 1: ci at end of r_i, cj at start of r_j
        if r_i and r_j:
            if r_i[-1] == ci and r_j[0] == cj:
                candidate = r_i + r_j
            elif r_i[-1] == cj and r_j[0] == ci:
                candidate = r_i + r_j[::-1]
            elif r_i[0] == ci and r_j[-1] == cj:
                candidate = r_i[::-1] + r_j
            elif r_i[0] == cj and r_j[-1] == ci:
                candidate = r_i[::-1] + r_j
            else:
                # Try inserting
                candidate = None
                # Try appending r_j to r_i in all orientations
                for orient_i in [r_i, r_i[::-1]]:
                    for orient_j in [r_j, r_j[::-1]]:
                        # Check if endpoints match
                        if orient_i[-1] in orient_j or orient_j[-1] in orient_i:
                            pass
                        # Just try concatenation
                        cand = orient_i + orient_j
                        if _route_feasible(cand, dist_matrix, instance, n_customers,
                                           stations_set, battery_max, speed,
                                           recharge_rate, horizon, capacity):
                            if merged is None or _compute_total_distance([cand], dist_matrix) < _compute_total_distance([merged], dist_matrix):
                                merged = cand
                            break
                    if merged:
                        break
                    if merged:
                        break
                if merged:
                    candidate = merged

        if candidate and _route_feasible(candidate, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed,
                                          recharge_rate, horizon, capacity):
            routes[route_i] = candidate
            routes[route_j] = []
            loads[route_i] += loads[route_j]
            union(ci, cj)

    return [r for r in routes if r]


def _nearest_neighbor_construct(instance, dist_matrix, n_customers, stations_set,
                                 battery_max, speed, recharge_rate, horizon, capacity,
                                 seed=0, n_routes=5):
    """Nearest neighbor construction with multiple restarts."""
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    customers = list(range(1, n_customers + 1))
    random.shuffle(customers)

    routes = []
    route_loads = []
    unvisited = set(customers)

    for _ in range(n_routes):
        if not unvisited:
            break
        # Pick starting customer
        start = random.choice(list(unvisited))
        route = [start]
        unvisited.remove(start)
        load = demands[start]
        current = start
        battery = battery_max
        time_at = 0.0

        # Travel from depot to start
        d = dist_matrix[0, start]
        time_at += d / speed
        battery -= d
        if time_at < ready[start]:
            time_at = ready[start]
        if time_at <= due[start] + EPS:
            time_at += instance["service"][start]

        while unvisited:
            best_next = None
            best_dist = float('inf')
            for cust in unvisited:
                if load + demands[cust] > capacity:
                    continue
                d = dist_matrix[current, cust]
                if battery - d < -EPS:
                    continue
                new_time = time_at + d / speed
                if new_time < ready[cust]:
                    new_time = ready[cust]
                if new_time > due[cust] + EPS:
                    continue
                if d < best_dist:
                    best_dist = d
                    best_next = cust
            if best_next is None:
                break
            route.append(best_next)
            unvisited.remove(best_next)
            load += demands[best_next]
            d = dist_matrix[current, best_next]
            time_at += d / speed
            battery -= d
            if time_at < ready[best_next]:
                time_at = ready[best_next]
            time_at += instance["service"][best_next]
            current = best_next

        # Check return to depot
        d = dist_matrix[current, 0]
        if battery - d >= -EPS and time_at + d / speed <= horizon + EPS:
            routes.append(route)
            route_loads.append(load)
        else:
            # Can't complete route, add back
            for c in route:
                unvisited.add(c)

    # Add any remaining customers as individual routes
    for cust in unvisited:
        routes.append([cust])
        route_loads.append(demands[cust])

    return routes


def _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                       battery_max, speed, recharge_rate, horizon, capacity, seed=0):
    random.seed(seed)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]

    customers = list(range(1, n_customers + 1))
    sort_key = seed % 4
    if sort_key == 0:
        customers.sort(key=lambda x: due[x])
    elif sort_key == 1:
        customers.sort(key=lambda x: dist_matrix[0, x])
    elif sort_key == 2:
        customers.sort(key=lambda x: due[x] - ready[x])
    else:
        random.shuffle(customers)

    routes = []
    route_loads = []

    for cust in customers:
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')

        for r_idx, route in enumerate(routes):
            if route_loads[r_idx] + demands[cust] > capacity:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [cust] + route[pos:]
                if _route_feasible(new_route, dist_matrix, instance, n_customers,
                                    stations_set, battery_max, speed, recharge_rate,
                                    horizon, capacity):
                    if pos == 0:
                        cost = dist_matrix[0, cust] + dist_matrix[cust, route[0]] - dist_matrix[0, route[0]]
                    elif pos == len(route):
                        cost = dist_matrix[route[-1], cust] + dist_matrix[cust, 0] - dist_matrix[route[-1], 0]
                    else:
                        cost = dist_matrix[route[pos-1], cust] + dist_matrix[cust, route[pos]] - dist_matrix[route[pos-1], route[pos]]
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos

        if best_route_idx == -1:
            routes.append([cust])
            route_loads.append(demands[cust])
        else:
            routes[best_route_idx].insert(best_pos, cust)
            route_loads[best_route_idx] += demands[cust]

    return routes


def _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                        battery_max, speed, recharge_rate, horizon, capacity,
                        n_clusters=5, seed=0):
    random.seed(seed)
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = instance["demands"]
    due = instance["due"]

    customer_coords = coords[1:n_customers+1]
    n_cust = n_customers

    centers = []
    remaining = list(range(n_cust))
    first = random.choice(remaining)
    centers.append(customer_coords[first].copy())
    remaining.remove(first)

    while len(centers) < n_clusters and remaining:
        max_dist = -1
        max_idx = 0
        for idx, cust_idx in enumerate(remaining):
            min_d = min(np.linalg.norm(customer_coords[cust_idx] - c) for c in centers)
            if min_d > max_dist:
                max_dist = min_d
                max_idx = idx
        centers.append(customer_coords[remaining[max_idx]].copy())
        remaining.pop(max_idx)

    clusters = [[] for _ in range(n_clusters)]
    for cust_idx in range(n_cust):
        min_d = float('inf')
        best_c = 0
        for c_idx, center in enumerate(centers):
            d = np.linalg.norm(customer_coords[cust_idx] - center)
            if d < min_d:
                min_d = d
                best_c = c_idx
        clusters[best_c].append(cust_idx + 1)

    routes = []
    for cluster in clusters:
        if not cluster:
            continue
        cluster.sort(key=lambda v: due[v])
        route = []
        for cust in cluster:
            test = route + [cust]
            if _route_feasible(test, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity):
                route = test
            else:
                if route:
                    routes.append(route)
                route = [cust]
        if route:
            routes.append(route)

    return routes


def _two_opt(route, dist_matrix, instance, n_customers, stations_set,
             battery_max, speed, recharge_rate, horizon, capacity, max_iters=100):
    if len(route) < 2:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                if is_feasible(new_route):
                    new_dist = route_dist(new_route)
                    if new_dist < best_dist - EPS:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
        if not improved:
            break

    return best_route


def _or_opt(route, dist_matrix, instance, n_customers, stations_set,
            battery_max, speed, recharge_rate, horizon, capacity, max_iters=50):
    if len(route) < 3:
        return route

    def is_feasible(r):
        return _route_feasible(r, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_dist(r):
        d = 0.0
        prev = 0
        for node in r:
            d += dist_matrix[prev, node]
            prev = node
        d += dist_matrix[prev, 0]
        return d

    best_route = route[:]
    best_dist = route_dist(best_route)

    for _ in range(max_iters):
        improved = False
        n = len(best_route)
        for seg_len in [1, 2, 3]:
            for i in range(n - seg_len + 1):
                segment = best_route[i:i+seg_len]
                for j in range(n - seg_len + 1):
                    if i == j or (j < i + seg_len and j + seg_len > i):
                        continue
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    if j > i:
                        j = j - seg_len
                    new_route = remaining[:j] + segment + remaining[j:]
                    if len(new_route) != len(best_route):
                        continue
                    if is_feasible(new_route):
                        new_dist = route_dist(new_route)
                        if new_dist < best_dist - EPS:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
        if not improved:
            break

    return best_route


def _local_search(routes, dist_matrix, instance, n_customers, stations_set,
                  battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)
    start_time = time.time()

    # Intra-route 2-opt and OR-opt
    for r_idx in range(len(routes)):
        if time.time() > deadline:
            break
        routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, max_iters=80)
        routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                 stations_set, battery_max, speed, recharge_rate,
                                 horizon, capacity, max_iters=40)
    current_dist = get_total_dist(routes)

    # Inter-route moves
    max_passes = 15
    for _pass in range(max_passes):
        if time.time() > deadline:
            break
        improved = False

        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - EPS:
                                routes = new_routes
                                current_dist = new_dist
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

    routes = [r for r in routes if len(r) > 0]
    return routes


def _simulated_annealing(routes, dist_matrix, instance, n_customers, stations_set,
                         battery_max, speed, recharge_rate, horizon, capacity, deadline):
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_routes = [r[:] for r in routes]
    current_dist = get_total_dist(current_routes)
    best_routes = [r[:] for r in current_routes]
    best_dist = current_dist

    start_temp = 10.0
    end_temp = 0.05
    max_iters = 4000
    cooling_rate = (end_temp / start_temp) ** (1.0 / max(1, max_iters))
    temp = start_temp

    start_time = time.time()
    for iteration in range(max_iters):
        if time.time() > deadline:
            break

        new_routes = [r[:] for r in current_routes]
        perturbation_type = random.random()

        if perturbation_type < 0.4:
            # Move a customer
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    cust = new_routes[r1][pos1]
                    if 1 <= cust <= n_customers:
                        new_routes[r1].pop(pos1)
                        valid_targets = [r for r in range(len(new_routes))
                                         if route_load(new_routes[r]) + demands[cust] <= capacity]
                        if valid_targets:
                            r2 = random.choice(valid_targets)
                            pos2 = random.randint(0, len(new_routes[r2]))
                            new_routes[r2].insert(pos2, cust)
        elif perturbation_type < 0.7:
            # Swap two customers
            if len(new_routes) >= 2:
                r1 = random.randint(0, len(new_routes) - 1)
                r2 = random.randint(0, len(new_routes) - 1)
                if r1 != r2 and len(new_routes[r1]) > 0 and len(new_routes[r2]) > 0:
                    pos1 = random.randint(0, len(new_routes[r1]) - 1)
                    pos2 = random.randint(0, len(new_routes[r2]) - 1)
                    cust1 = new_routes[r1][pos1]
                    cust2 = new_routes[r2][pos2]
                    if 1 <= cust1 <= n_customers and 1 <= cust2 <= n_customers:
                        load1 = route_load(new_routes[r1])
                        load2 = route_load(new_routes[r2])
                        if load1 - demands[cust1] + demands[cust2] <= capacity and \
                           load2 - demands[cust2] + demands[cust1] <= capacity:
                            new_routes[r1][pos1] = cust2
                            new_routes[r2][pos2] = cust1
        else:
            # Reverse segment
            if len(new_routes) > 0:
                r1 = random.randint(0, len(new_routes) - 1)
                if len(new_routes[r1]) >= 2:
                    i = random.randint(0, len(new_routes[r1]) - 2)
                    j = random.randint(i + 1, len(new_routes[r1]) - 1)
                    new_routes[r1][i:j+1] = new_routes[r1][i:j+1][::-1]

        new_routes = [r for r in new_routes if len(r) > 0]
        all_feasible = True
        served = set()
        for route in new_routes:
            if not is_feasible(route):
                all_feasible = False
                break
            for node in route:
                if 1 <= node <= n_customers:
                    if node in served:
                        all_feasible = False
                        break
                    served.add(node)
            if not all_feasible:
                break

        if all_feasible and served == set(range(1, n_customers + 1)):
            new_dist = get_total_dist(new_routes)
            delta = new_dist - current_dist
            if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-10)):
                current_routes = new_routes
                current_dist = new_dist
                if current_dist < best_dist:
                    best_routes = [r[:] for r in current_routes]
                    best_dist = current_dist

        temp *= cooling_rate

    return best_routes


def _polish_routes(routes, dist_matrix, instance, n_customers, stations_set,
                   battery_max, speed, recharge_rate, horizon, capacity, deadline):
    """Final high-precision polish: multiple passes of 2-opt + OR-opt + inter-route moves."""
    demands = instance["demands"]

    def get_total_dist(rts):
        return _compute_total_distance(rts, dist_matrix)

    def is_feasible(route):
        return _route_feasible(route, dist_matrix, instance, n_customers, stations_set,
                                battery_max, speed, recharge_rate, horizon, capacity)

    def route_load(route):
        return sum(demands[node] for node in route if 1 <= node <= n_customers)

    current_dist = get_total_dist(routes)

    for _pass in range(3):
        if time.time() > deadline:
            break
        for r_idx in range(len(routes)):
            if time.time() > deadline:
                break
            routes[r_idx] = _two_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                      stations_set, battery_max, speed, recharge_rate,
                                      horizon, capacity, max_iters=50)
            routes[r_idx] = _or_opt(routes[r_idx], dist_matrix, instance, n_customers,
                                     stations_set, battery_max, speed, recharge_rate,
                                     horizon, capacity, max_iters=30)

        # Inter-route moves
        for r1 in range(len(routes)):
            if time.time() > deadline:
                break
            for pos1 in range(len(routes[r1])):
                if time.time() > deadline:
                    break
                cust = routes[r1][pos1]
                if not (1 <= cust <= n_customers):
                    continue
                for r2 in range(len(routes)):
                    if time.time() > deadline:
                        break
                    if r1 == r2:
                        continue
                    if route_load(routes[r2]) + demands[cust] > capacity:
                        continue
                    for pos2 in range(len(routes[r2]) + 1):
                        if time.time() > deadline:
                            break
                        route1_new = routes[r1][:pos1] + routes[r1][pos1+1:]
                        route2_new = routes[r2][:pos2] + [cust] + routes[r2][pos2:]
                        if is_feasible(route1_new) and is_feasible(route2_new):
                            new_routes = [r[:] for r in routes]
                            new_routes[r1] = route1_new
                            new_routes[r2] = route2_new
                            new_dist = get_total_dist(new_routes)
                            if new_dist < current_dist - EPS:
                                routes = new_routes
                                current_dist = new_dist
                                break
                    if time.time() > deadline:
                        break

    routes = [r for r in routes if len(r) > 0]
    return routes


def plan(instance):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    coords = np.array(instance["coords"], dtype=np.float64)
    dist_matrix = _build_dist_matrix(coords)

    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    capacity = instance["capacity"]

    best_routes = None
    best_dist = float('inf')

    # Phase 1: Generate diverse initial solutions
    # Greedy constructions with different sort keys
    for i in range(8):
        if time.time() > deadline - 40:
            break
        routes = _greedy_construct(instance, dist_matrix, n_customers, stations_set,
                                    battery_max, speed, recharge_rate, horizon, capacity,
                                    seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Savings-based construction
    for i in range(3):
        if time.time() > deadline - 35:
            break
        routes = _savings_construct(instance, dist_matrix, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity,
                                     seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Nearest neighbor constructions
    for i in range(3):
        if time.time() > deadline - 30:
            break
        routes = _nearest_neighbor_construct(instance, dist_matrix, n_customers, stations_set,
                                              battery_max, speed, recharge_rate, horizon, capacity,
                                              seed=i, n_routes=5)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    # Cluster-based constructions
    for i in range(4):
        if time.time() > deadline - 25:
            break
        routes = _cluster_construct(instance, dist_matrix, n_customers, stations_set,
                                     battery_max, speed, recharge_rate, horizon, capacity,
                                     n_clusters=4 + i, seed=i)
        valid, dist = _validate_solution(routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = [r[:] for r in routes]

    if best_routes is None:
        best_routes = [[v] for v in range(1, n_customers + 1)]
        best_dist = _compute_total_distance(best_routes, dist_matrix)

    # Phase 2: Local search
    if time.time() < deadline - 25:
        improved = _local_search(best_routes, dist_matrix, instance, n_customers,
                                  stations_set, battery_max, speed, recharge_rate,
                                  horizon, capacity, deadline - 25)
        valid, dist = _validate_solution(improved, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = improved

    # Phase 3: Simulated annealing
    if time.time() < deadline - 18:
        sa_routes = _simulated_annealing(best_routes, dist_matrix, instance, n_customers,
                                          stations_set, battery_max, speed, recharge_rate,
                                          horizon, capacity, deadline - 12)
        valid, dist = _validate_solution(sa_routes, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = sa_routes

    # Phase 4: Polish
    if time.time() < deadline - 5:
        polished = _polish_routes(best_routes, dist_matrix, instance, n_customers,
                                   stations_set, battery_max, speed, recharge_rate,
                                   horizon, capacity, deadline - 2)
        valid, dist = _validate_solution(polished, instance)
        if valid and dist < best_dist:
            best_dist = dist
            best_routes = polished

    # Final validation
    valid, dist = _validate_solution(best_routes, instance)
    if not valid:
        best_routes = [[v] for v in range(1, n_customers + 1)]

    return best_routes
