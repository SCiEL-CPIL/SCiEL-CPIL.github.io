"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time
import numpy as np


def _build_dist_matrix(coords):
    n = len(coords)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = math.hypot(dx, dy)
            mat[i, j] = d
            mat[j, i] = d
    return mat


def _simulate_route(route, instance, dist):
    """
    Simulate a route (list of customer/station node ids, depot implicit at ends).
    Returns (feasible, total_distance, end_time, end_battery).
    """
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])

    total_dist = 0.0
    total_demand = 0.0
    battery = battery_cap
    t = 0.0
    prev_node = 0  # depot

    # Check capacity
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return (False, total_dist, t, battery)

    for v in route:
        d = dist[prev_node][v]
        t += d / speed
        battery -= d

        if battery < -1e-9:
            return (False, total_dist, t, battery)

        if v <= n_customers:
            # Customer
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return (False, total_dist, t, battery)
            t += service[v]
        else:
            # Station
            energy_needed = battery_cap - battery
            t += energy_needed * recharge_rate
            battery = battery_cap

        total_dist += d
        prev_node = v

    # Return to depot
    d = dist[prev_node][0]
    t += d / speed
    battery -= d
    total_dist += d

    if battery < -1e-9:
        return (False, total_dist, t, battery)
    if t > horizon + 1e-9:
        return (False, total_dist, t, battery)

    return (True, total_dist, t, battery)


def _compute_total_distance(routes, dist):
    total = 0.0
    for route in routes:
        if not route:
            continue
        # depot -> first
        total += dist[0][route[0]]
        # between stops
        for i in range(len(route) - 1):
            total += dist[route[i]][route[i + 1]]
        # last -> depot
        total += dist[route[-1]][0]
    return total


def _validate_all(routes, instance, dist):
    """Check all constraints for all routes."""
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations_set = set(instance["stations"])

    served = set()
    for route in routes:
        total_demand = 0.0
        battery = battery_cap
        t = 0.0
        prev_node = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev_node][v]
            t += d / speed
            battery -= d
            if battery < -1e-9:
                return False
            if v <= n_customers:
                total_demand += demands[v]
                if total_demand > capacity + 1e-9:
                    return False
                t = max(t, ready[v])
                if t > due[v] + 1e-9:
                    return False
                t += service[v]
                if v in served:
                    return False
                served.add(v)
            else:
                if v not in stations_set:
                    return False
                energy_needed = battery_cap - battery
                t += energy_needed * recharge_rate
                battery = battery_cap
            total_dist += d
            prev_node = v

        d = dist[prev_node][0]
        t += d / speed
        battery -= d
        total_dist += d
        if battery < -1e-9:
            return False
        if t > horizon + 1e-9:
            return False

    # All customers served exactly once
    for c in range(1, n_customers + 1):
        if c not in served:
            return False
    return True


def _insert_station_if_needed(route, instance, dist):
    """
    Given a route of customers only, try to insert stations to make it feasible.
    Returns the modified route or None if infeasible.
    """
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]

    # Check capacity first
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity + 1e-9:
        return None

    # Simulate without stations to find where battery goes negative
    # Strategy: insert a station after each customer if battery is getting low
    # More aggressive: pre-insert stations at points where battery would drop below a threshold

    # Try inserting stations greedily
    result = list(route)
    max_stations = 10  # limit to avoid infinite loops

    for _ in range(max_stations):
        feasible, total_dist, end_t, end_batt = _simulate_route(result, instance, dist)
        if feasible:
            return result

        # Find the point where battery first goes negative
        battery = battery_cap
        t = 0.0
        prev_node = 0
        insert_pos = -1
        insert_station = -1

        for i, v in enumerate(result):
            d = dist[prev_node][v]
            battery -= d
            if battery < 0:
                # Need a station before this stop
                # Find best station to insert at position i
                best_station = None
                best_cost = float('inf')
                for st in stations:
                    # Insert st between prev_node and v
                    d1 = dist[prev_node][st]
                    d2 = dist[st][v]
                    # Check if battery can reach station
                    if battery_cap - d1 >= -1e-9:
                        cost = d1 + d2
                        if cost < best_cost:
                            best_cost = cost
                            best_station = st
                if best_station is not None:
                    result.insert(i, best_station)
                    return _insert_station_if_needed(result, instance, dist)
                else:
                    return None
            prev_node = v

        # Also check return to depot
        d = dist[prev_node][0]
        battery -= d
        if battery < 0:
            # Need station before returning
            best_station = None
            best_cost = float('inf')
            for st in stations:
                d1 = dist[prev_node][st]
                d2 = dist[st][0]
                if battery_cap - d1 >= -1e-9:
                    cost = d1 + d2
                    if cost < best_cost:
                        best_cost = cost
                        best_station = st
            if best_station is not None:
                result.append(best_station)
                return _insert_station_if_needed(result, instance, dist)
            else:
                return None

        return result

    return None


def _clarke_wright_savings(customers, instance, dist):
    """
    Clarke-Wright savings heuristic to build initial routes.
    Returns list of routes (each a list of customer ids).
    """
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    demands = instance["demands"]

    # Start with each customer as its own route
    routes = [[c] for c in customers]
    route_of = {c: i for i, c in enumerate(customers)}
    route_load = [demands[c] for c in customers]

    # Compute savings
    savings = []
    for i in range(len(customers)):
        for j in range(i + 1, len(customers)):
            ci = customers[i]
            cj = customers[j]
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    # Union-Find for connectivity within a route
    parent = list(range(len(customers)))
    rank = [0] * len(customers)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for s, i, j in savings:
        if s <= 0:
            continue
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # Check capacity
        if route_load[ri] + route_load[rj] > capacity + 1e-9:
            continue
        # Merge: we need to merge the two routes
        # Find the routes
        route_i = routes[ri]
        route_j = routes[rj]

        # Try to merge at the ends
        # The two customers that were endpoints: customers[i] is in route_i, customers[j] is in route_j
        # We need to check if they are at the ends of their respective routes
        ci = customers[i]
        cj = customers[j]

        # Find positions
        pos_i = route_i.index(ci)
        pos_j = route_j.index(cj)

        # For savings to work, ci should be at an end of route_i and cj at an end of route_j
        # Let's just do a simple merge: concatenate
        # Determine orientation
        # We want ci and cj to be adjacent
        # Reverse route_i if ci is at the start
        if pos_i == 0:
            route_i = list(reversed(route_i))
        # Reverse route_j if cj is at the end
        if pos_j == len(route_j) - 1:
            route_j = list(reversed(route_j))

        # Now ci should be at end of route_i, cj at start of route_j
        if route_i[-1] != ci or route_j[0] != cj:
            # Can't merge easily, skip
            continue

        merged = route_i + route_j
        routes[ri] = merged
        routes[rj] = []
        route_load[ri] += route_load[rj]

    # Remove empty routes
    routes = [r for r in routes if r]
    return routes


def _two_opt_route(route, instance, dist, max_iter=50):
    """
    2-opt improvement on a single route (customers only, then add stations).
    """
    best_route = list(route)
    best_dist = _compute_total_distance([best_route], dist)
    improved = True
    iters = 0

    while improved and iters < max_iter:
        improved = False
        iters += 1
        n = len(best_route)
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i..j
                new_route = best_route[:i] + list(reversed(best_route[i:j + 1])) + best_route[j + 1:]
                new_dist = _compute_total_distance([new_route], dist)
                if new_dist < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = new_dist
                    improved = True

    return best_route


def _or_opt_route(route, instance, dist, max_iter=30):
    """
    Or-opt: move 1 or 2 consecutive customers to another position.
    """
    best_route = list(route)
    best_dist = _compute_total_distance([best_route], dist)
    improved = True
    iters = 0

    while improved and iters < max_iter:
        improved = False
        iters += 1
        n = len(best_route)
        for length in [1, 2]:
            for i in range(n - length + 1):
                segment = best_route[i:i + length]
                # Remove segment
                remaining = best_route[:i] + best_route[i + length:]
                for j in range(len(remaining) + 1):
                    if j == i:
                        continue
                    new_route = remaining[:j] + segment + remaining[j:]
                    new_dist = _compute_total_distance([new_route], dist)
                    if new_dist < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    return best_route


def _relocate_customer(routes, instance, dist):
    """
    Try to move a customer from one route to another.
    Returns modified routes or None.
    """
    capacity = instance["capacity"]
    demands = instance["demands"]

    for i in range(len(routes)):
        for j in range(len(routes[i])):
            c = routes[i][j]
            # Try inserting into other routes
            for k in range(len(routes)):
                if k == i:
                    continue
                if demands[c] + sum(demands[x] for x in routes[k]) > capacity + 1e-9:
                    continue
                for pos in range(len(routes[k]) + 1):
                    new_route_k = routes[k][:pos] + [c] + routes[k][pos:]
                    # Check if this improves total distance
                    old_dist_i = _compute_total_distance([routes[i]], dist)
                    old_dist_k = _compute_total_distance([routes[k]], dist)
                    new_dist_i = _compute_total_distance([routes[i][:j] + routes[i][j + 1:]], dist)
                    new_dist_k = _compute_total_distance([new_route_k], dist)
                    if new_dist_i + new_dist_k < old_dist_i + old_dist_k - 1e-9:
                        # Perform move
                        routes[i] = routes[i][:j] + routes[i][j + 1:]
                        routes[k] = new_route_k
                        return True
    return False


def _swap_customers(routes, instance, dist):
    """
    Try to swap two customers from different routes.
    """
    capacity = instance["capacity"]
    demands = instance["demands"]

    for i in range(len(routes)):
        for j in range(len(routes[i])):
            ci = routes[i][j]
            for k in range(i + 1, len(routes)):
                for l in range(len(routes[k])):
                    ck = routes[k][l]
                    # Check capacity after swap
                    load_i = sum(demands[x] for x in routes[i])
                    load_k = sum(demands[x] for x in routes[k])
                    new_load_i = load_i - demands[ci] + demands[ck]
                    new_load_k = load_k - demands[ck] + demands[ci]
                    if new_load_i > capacity + 1e-9 or new_load_k > capacity + 1e-9:
                        continue
                    # Try swap
                    routes[i][j], routes[k][l] = routes[k][l], routes[i][j]
                    old_total = _compute_total_distance(routes, dist)
                    new_total = _compute_total_distance(routes, dist)
                    if new_total < old_total - 1e-9:
                        return True
                    else:
                        routes[i][j], routes[k][l] = routes[i][j], routes[k][l]
    return False


def _local_search(routes, instance, dist, max_rounds=10):
    """
    Apply local search moves: 2-opt, or-opt, relocate, swap.
    """
    for _ in range(max_rounds):
        improved = False

        # 2-opt on each route
        for i in range(len(routes)):
            if len(routes[i]) >= 2:
                new_route = _two_opt_route(routes[i], instance, dist, max_iter=20)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

        # Or-opt on each route
        for i in range(len(routes)):
            if len(routes[i]) >= 2:
                new_route = _or_opt_route(routes[i], instance, dist, max_iter=10)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

        # Relocate
        if _relocate_customer(routes, instance, dist):
            improved = True

        # Swap
        if _swap_customers(routes, instance, dist):
            improved = True

        if not improved:
            break

    return routes


def _greedy_construct(customers, instance, dist):
    """
    Greedy construction: insert customers one by one into the best position.
    """
    capacity = instance["capacity"]
    demands = instance["demands"]

    # Sort by due time
    customers_sorted = sorted(customers, key=lambda c: instance["due"][c])

    routes = []
    route_loads = []

    for c in customers_sorted:
        best_pos = None
        best_cost = float('inf')
        best_route_idx = -1

        # Try adding to existing routes
        for i in range(len(routes)):
            if route_loads[i] + demands[c] > capacity + 1e-9:
                continue
            for pos in range(len(routes[i]) + 1):
                new_route = routes[i][:pos] + [c] + routes[i][pos:]
                cost = _compute_total_distance([new_route], dist)
                old_cost = _compute_total_distance([routes[i]], dist)
                delta = cost - old_cost
                if delta < best_cost:
                    best_cost = delta
                    best_pos = pos
                    best_route_idx = i

        # Try creating a new route
        new_cost = _compute_total_distance([[c]], dist)
        if best_route_idx == -1 or new_cost < best_cost:
            routes.append([c])
            route_loads.append(demands[c])
        else:
            routes[best_route_idx].insert(best_pos, c)
            route_loads[best_route_idx] += demands[c]

    return routes


def _genetic_algorithm(customers, instance, dist, time_budget=60):
    """
    Genetic algorithm for EV routing.
    """
    capacity = instance["capacity"]
    demands = instance["demands"]
    n_customers = instance["n_customers"]

    start_time = time.time()
    pop_size = 20
    elite_size = 5
    mutation_rate = 0.3
    crossover_rate = 0.7

    # Initial population: mix of heuristics
    population = []

    # Greedy
    pop1 = _greedy_construct(customers, instance, dist)
    population.append(pop1)

    # Clarke-Wright
    pop2 = _clarke_wright_savings(customers, instance, dist)
    population.append(pop2)

    # Random permutations split into routes by capacity
    for _ in range(pop_size - 2):
        random.shuffle(customers)
        routes = []
        current = []
        current_load = 0
        for c in customers:
            if current_load + demands[c] <= capacity:
                current.append(c)
                current_load += demands[c]
            else:
                if current:
                    routes.append(current)
                current = [c]
                current_load = demands[c]
        if current:
            routes.append(current)
        population.append(routes)

    # Evaluate and keep best
    def evaluate(routes):
        total_dist = _compute_total_distance(routes, dist)
        return total_dist

    # Local search on initial population
    for i in range(len(population)):
        population[i] = _local_search(population[i], instance, dist, max_rounds=3)

    best_routes = min(population, key=evaluate)
    best_dist = evaluate(best_routes)

    generation = 0
    while time.time() - start_time < time_budget:
        generation += 1

        # Sort by fitness
        population.sort(key=evaluate)
        best_routes = population[0]
        best_dist = evaluate(best_routes)

        # Selection: keep elites
        new_population = [list(r) for r in population[:elite_size]]

        # Generate offspring
        while len(new_population) < pop_size:
            # Select parents
            p1 = random.choice(population[:elite_size * 2])
            p2 = random.choice(population[:elite_size * 2])

            if random.random() < crossover_rate and len(p1) > 0 and len(p2) > 0:
                # Crossover: order-based
                # Flatten both
                flat1 = [c for r in p1 for c in r]
                flat2 = [c for r in p2 for c in r]

                # Pick a random subset of positions from p1
                mask = [random.random() < 0.5 for _ in range(len(flat1))]
                child_flat = []
                for i, m in enumerate(mask):
                    if m and i < len(flat1):
                        child_flat.append(flat1[i])

                # Fill remaining from p2
                for c in flat2:
                    if c not in child_flat:
                        child_flat.append(c)

                # Split into routes by capacity
                child_routes = []
                current = []
                current_load = 0
                for c in child_flat:
                    if current_load + demands[c] <= capacity:
                        current.append(c)
                        current_load += demands[c]
                    else:
                        if current:
                            child_routes.append(current)
                        current = [c]
                        current_load = demands[c]
                if current:
                    child_routes.append(current)
            else:
                # Copy parent
                child_routes = [list(r) for r in random.choice([p1, p2])]

            # Mutation
            if random.random() < mutation_rate:
                # Random swap
                if len(child_routes) >= 2:
                    ri = random.randint(0, len(child_routes) - 1)
                    rj = random.randint(0, len(child_routes) - 1)
                    if ri != rj and len(child_routes[ri]) > 0 and len(child_routes[rj]) > 0:
                        ii = random.randint(0, len(child_routes[ri]) - 1)
                        jj = random.randint(0, len(child_routes[rj]) - 1)
                        c1 = child_routes[ri][ii]
                        c2 = child_routes[rj][jj]
                        # Check capacity
                        load_ri = sum(demands[x] for x in child_routes[ri])
                        load_rj = sum(demands[x] for x in child_routes[rj])
                        new_load_ri = load_ri - demands[c1] + demands[c2]
                        new_load_rj = load_rj - demands[c2] + demands[c1]
                        if new_load_ri <= capacity and new_load_rj <= capacity:
                            child_routes[ri][ii] = c2
                            child_routes[rj][jj] = c1
                # Random relocate
                if len(child_routes) >= 2:
                    ri = random.randint(0, len(child_routes) - 1)
                    rj = random.randint(0, len(child_routes) - 1)
                    if ri != rj and len(child_routes[ri]) > 1:
                        ii = random.randint(0, len(child_routes[ri]) - 1)
                        c = child_routes[ri][ii]
                        load_rj = sum(demands[x] for x in child_routes[rj])
                        if load_rj + demands[c] <= capacity:
                            child_routes[ri].pop(ii)
                            jj = random.randint(0, len(child_routes[rj]))
                            child_routes[rj].insert(jj, c)

            new_population.append(child_routes)

        # Local search on some offspring
        for i in range(len(new_population)):
            if random.random() < 0.5:
                new_population[i] = _local_search(new_population[i], instance, dist, max_rounds=2)

        population = new_population

    return population[0]


def plan(instance):
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill

    coords = instance["coords"]
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    dist = _build_dist_matrix(coords)

    # Generate initial solutions using different heuristics
    candidates = []

    # 1. One customer per route (baseline)
    candidates.append([[c] for c in customers])

    # 2. Greedy construction
    try:
        greedy = _greedy_construct(customers, instance, dist)
        candidates.append(greedy)
    except:
        pass

    # 3. Clarke-Wright savings
    try:
        cw = _clarke_wright_savings(customers, instance, dist)
        candidates.append(cw)
    except:
        pass

    # 4. Local search on each candidate
    for i in range(len(candidates)):
        try:
            candidates[i] = _local_search(candidates[i], instance, dist, max_rounds=5)
        except:
            pass

    # 5. Genetic algorithm if time permits
    if time.time() - start_time < deadline - 30:
        try:
            ga_result = _genetic_algorithm(customers, instance, dist, time_budget=deadline - 35 - (time.time() - start_time))
            candidates.append(ga_result)
        except:
            pass

    # 6. Additional local search on best candidates
    best_dist = float('inf')
    best_routes = None
    for routes in candidates:
        total_dist = _compute_total_distance(routes, dist)
        if total_dist < best_dist:
            best_dist = total_dist
            best_routes = [list(r) for r in routes]

    # Final local search on best
    if best_routes is not None and time.time() - start_time < deadline - 10:
        try:
            best_routes = _local_search(best_routes, instance, dist, max_rounds=10)
        except:
            pass

    # Now add stations to make routes feasible
    final_routes = []
    for route in best_routes:
        if not route:
            continue
        feasible_route = _insert_station_if_needed(route, instance, dist)
        if feasible_route is not None:
            final_routes.append(feasible_route)
        else:
            # Fallback: one customer per route
            for c in route:
                single = [c]
                feasible_single = _insert_station_if_needed(single, instance, dist)
                if feasible_single is not None:
                    final_routes.append(feasible_single)
                else:
                    final_routes.append(single)

    # Validate
    if not _validate_all(final_routes, instance, dist):
        # Fallback to one-per-route
        fallback = []
        for c in customers:
            single = [c]
            feasible_single = _insert_station_if_needed(single, instance, dist)
            if feasible_single is not None:
                fallback.append(feasible_single)
            else:
                fallback.append(single)
        return fallback

    return final_routes

# EVOLVE-BLOCK-END
