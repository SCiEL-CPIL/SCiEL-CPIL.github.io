import numpy as np
import time
import math
import random


def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin vs 150s wall-clock kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = list(instance["stations"])
    station_set = set(stations)
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dx = coords[:, 0]
    dy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        dist[i, :] = np.sqrt((dx[i] - dx)**2 + (dy[i] - dy)**2)

    customers = list(range(1, n_customers + 1))
    customer_set = set(customers)

    def simulate_route(route):
        """
        Simulate a route (list of node ids) starting from depot (node 0).
        Returns (total_distance, feasible, final_time, final_battery) or None if infeasible.
        """
        if not route:
            return (0.0, True, 0.0, battery_cap)

        t = 0.0
        battery = battery_cap
        total_dist = 0.0
        current = 0  # depot

        for node in route:
            d = dist[current, node]
            t += d / speed
            battery -= d
            total_dist += d

            if battery < -1e-9:
                return None  # battery went negative

            if node in customer_set:
                # Customer service
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return None  # missed due time
                t += service[node]
            elif node in station_set:
                # Charging station
                recharge_time = (battery_cap - battery) * recharge_rate
                t += recharge_time
                battery = battery_cap

            current = node

        # Return to depot
        d = dist[current, 0]
        t += d / speed
        battery -= d
        total_dist += d

        if battery < -1e-9:
            return None
        if t > horizon + 1e-9:
            return None

        return (total_dist, True, t, battery)

    def route_demand(route):
        return sum(demands[v] for v in route if v in customer_set)

    def total_distance(routes):
        total = 0.0
        for route in routes:
            sim = simulate_route(route)
            if sim is None:
                return float('inf')
            total += sim[0]
        return total

    def best_insertion_cost(route, node, allow_station=True):
        """
        Find the best position to insert node into route, returning (cost, new_route) or (inf, None).
        Tries all positions and also tries adding a station before the insertion point.
        """
        best_cost = float('inf')
        best_route = None
        base_demand = route_demand(route)

        if base_demand + demands[node] > capacity:
            return (float('inf'), None)

        # Try inserting at each position without station
        for pos in range(len(route) + 1):
            new_route = list(route[:pos]) + [node] + list(route[pos:])
            sim = simulate_route(new_route)
            if sim is not None:
                if sim[0] < best_cost:
                    best_cost = sim[0]
                    best_route = new_route

        # Try inserting with a station before the node
        if allow_station:
            for pos in range(len(route) + 1):
                for st in stations:
                    new_route = list(route[:pos]) + [st, node] + list(route[pos:])
                    sim = simulate_route(new_route)
                    if sim is not None:
                        if sim[0] < best_cost:
                            best_cost = sim[0]
                            best_route = new_route

        return (best_cost, best_route)

    def savings_based_construction():
        """
        Build initial solution using Clarke-Wright savings heuristic.
        """
        # Start with each customer in its own route
        routes = {c: [c] for c in customers}

        # Compute savings for merging two routes
        # Savings = dist(0, i) + dist(0, j) - dist(i, j)
        # But we need to be smarter with time windows and battery

        # Sort customers by due time for better initial ordering
        sorted_customers = sorted(customers, key=lambda v: due[v])

        # Greedy insertion approach: build routes by inserting customers
        unvisited = set(customers)
        built_routes = []

        for c in sorted_customers:
            if c not in unvisited:
                continue

            # Try to insert c into existing routes
            best_route_idx = -1
            best_cost = float('inf')
            best_new_route = None

            for i, route in enumerate(built_routes):
                cost, new_route = best_insertion_cost(route, c)
                if new_route is not None and cost < best_cost:
                    best_cost = cost
                    best_route_idx = i
                    best_new_route = new_route

            if best_route_idx >= 0:
                built_routes[best_route_idx] = best_new_route
            else:
                # Start new route
                built_routes.append([c])

            unvisited.remove(c)

        return built_routes

    def two_opt(route, max_iter=50):
        """
        Apply 2-opt improvement to a single route, maintaining feasibility.
        """
        if len(route) < 2:
            return route

        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = list(route[:i]) + list(reversed(route[i:j+1])) + list(route[j+1:])
                    if route_demand(new_route) > capacity:
                        continue
                    sim_new = simulate_route(new_route)
                    if sim_new is None:
                        continue
                    sim_old = simulate_route(route)
                    if sim_old is None:
                        continue
                    if sim_new[0] < sim_old[0] - 1e-6:
                        route = new_route
                        improved = True
            if time.time() > deadline - 5.0:
                break

        return route

    def or_opt(route, max_iter=30):
        """
        Apply Or-opt (relocate 1-3 consecutive nodes) improvement.
        """
        if len(route) < 3:
            return route

        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            n = len(route)
            for length in [1, 2, 3]:
                for i in range(n - length + 1):
                    segment = route[i:i+length]
                    # Remove segment
                    remaining = list(route[:i]) + list(route[i+length:])
                    if not remaining:
                        continue
                    # Try inserting segment at each position in remaining
                    for j in range(len(remaining) + 1):
                        new_route = list(remaining[:j]) + segment + list(remaining[j:])
                        if new_route == route:
                            continue
                        if route_demand(new_route) > capacity:
                            continue
                        sim_new = simulate_route(new_route)
                        if sim_new is None:
                            continue
                        sim_old = simulate_route(route)
                        if sim_old is None:
                            continue
                        if sim_new[0] < sim_old[0] - 1e-6:
                            route = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if time.time() > deadline - 5.0:
                break

        return route

    def local_search(routes, max_iters=100):
        """
        Combined local search: 2-opt, Or-opt, relocate, swap.
        """
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break

            improved = False

            # 2-opt
            for i in range(len(routes)):
                new_route = two_opt(routes[i], 20)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # Or-opt
            for i in range(len(routes)):
                new_route = or_opt(routes[i], 15)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True

            # Relocate moves
            for i in range(len(routes)):
                for pos_i, c in enumerate(routes[i]):
                    # Try moving c to each other route
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        if route_demand(routes[j]) + demands[c] > capacity:
                            continue
                        # Remove c from route_i
                        new_route_i = list(routes[i][:pos_i]) + list(routes[i][pos_i+1:])
                        # Try inserting c at various positions in route_j
                        cost, new_route_j = best_insertion_cost(routes[j], c)
                        if new_route_j is None:
                            continue
                        sim_i = simulate_route(new_route_i) if new_route_i else (0.0, True, 0.0, battery_cap)
                        sim_j = simulate_route(new_route_j)
                        if sim_i is None or sim_j is None:
                            continue
                        old_dist = simulate_route(routes[i])[0] + simulate_route(routes[j])[0]
                        new_dist = sim_i[0] + sim_j[0]
                        if new_dist < old_dist - 1e-6:
                            routes[i] = new_route_i
                            routes[j] = new_route_j
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Remove empty routes
            routes = [r for r in routes if r]

            if not improved:
                break

        return routes

    def lns_reoptimize(routes, removed_customers):
        """
        Large Neighborhood Search: re-optimize insertion of removed customers.
        """
        # Remove the customers from their routes
        for i in range(len(routes)):
            route = routes[i]
            new_route = [c for c in route if c not in removed_customers]
            routes[i] = new_route

        # Remove empty routes
        routes = [r for r in routes if r]

        # Re-insert removed customers using best insertion
        for c in removed_customers:
            best_route_idx = -1
            best_cost = float('inf')
            best_new_route = None

            for i, route in enumerate(routes):
                cost, new_route = best_insertion_cost(route, c)
                if new_route is not None and cost < best_cost:
                    best_cost = cost
                    best_route_idx = i
                    best_new_route = new_route

            if best_route_idx >= 0:
                routes[best_route_idx] = best_new_route
            else:
                # Start new route
                routes.append([c])

        return routes

    # Initial construction
    routes = savings_based_construction()

    # Local search
    routes = local_search(routes, 50)
    best_routes = [list(r) for r in routes]
    best_dist = total_distance(best_routes)

    # LNS iterations
    rng = random.Random(42)
    for lns_iter in range(200):
        if time.time() > deadline - 5.0:
            break

        # Randomly remove a subset of customers
        n_remove = rng.randint(3, min(8, len(customers)))
        removed = set(rng.sample(customers, n_remove))

        # Create a copy and re-optimize
        routes_copy = [list(r) for r in best_routes]
        routes_copy = lns_reoptimize(routes_copy, removed)

        # Local search on the new solution
        routes_copy = local_search(routes_copy, 20)

        d = total_distance(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d

    # Try different initial constructions if time permits
    if time.time() < deadline - 10.0:
        for _ in range(3):
            random_order = customers[:]
            rng.shuffle(random_order)

            # Greedy insertion with random order
            unvisited = set(customers)
            built_routes = []
            for c in random_order:
                if c not in unvisited:
                    continue
                best_route_idx = -1
                best_cost = float('inf')
                best_new_route = None
                for i, route in enumerate(built_routes):
                    cost, new_route = best_insertion_cost(route, c)
                    if new_route is not None and cost < best_cost:
                        best_cost = cost
                        best_route_idx = i
                        best_new_route = new_route
                if best_route_idx >= 0:
                    built_routes[best_route_idx] = best_new_route
                else:
                    built_routes.append([c])
                unvisited.remove(c)

            built_routes = local_search(built_routes, 30)
            d2 = total_distance(built_routes)
            if d2 < best_dist - 1e-6:
                best_routes = [list(r) for r in built_routes]
                best_dist = d2

    # Final validation
    all_served = []
    for route in best_routes:
        for c in route:
            if c in customer_set:
                all_served.append(c)

    if len(all_served) != len(customers) or len(set(all_served)) != len(customers):
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]

    return best_routes
