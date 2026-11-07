"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    customers = list(range(1, n_customers + 1))
    station_set = set(stations)
    depot = 0

    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))

    def route_distance(route):
        """Total Euclidean distance of a route including depot legs."""
        if not route:
            return 0.0
        total = dist_mat[depot, route[0]]
        for i in range(len(route) - 1):
            total += dist_mat[route[i], route[i + 1]]
        total += dist_mat[route[-1], depot]
        return total

    def simulate_route(route):
        """
        Simulate a route and return (feasible, time, battery, total_distance).
        Handles charging at stations.
        """
        if not route:
            return True, 0.0, battery, 0.0

        t = 0.0
        bat = battery
        total_dist = 0.0

        # Add depot at start
        nodes = [depot] + list(route) + [depot]

        for i in range(len(nodes) - 1):
            curr = nodes[i]
            next_node = nodes[i + 1]
            d = dist_mat[curr, next_node]

            # Check if battery is sufficient for this leg
            if bat < d:
                return False, t, bat, total_dist

            bat -= d
            t += d / speed
            total_dist += d

            # Process next node
            if next_node == depot:
                # Arrived at depot
                if t > horizon:
                    return False, t, bat, total_dist
                continue
            elif next_node in station_set:
                # Charge at station
                # Time to recharge: (battery - bat) * recharge_rate
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = battery
                if t > horizon:
                    return False, t, bat, total_dist
            else:
                # Customer
                if t < ready[next_node]:
                    t = ready[next_node]
                if t > due[next_node]:
                    return False, t, bat, total_dist
                t += service[next_node]
                if t > horizon:
                    return False, t, bat, total_dist

        return True, t, bat, total_dist

    def check_route_feasibility(route):
        feasible, t, bat, total_dist = simulate_route(route)
        return feasible

    def can_insert(route, customer, position):
        """Check if inserting customer at position in route is feasible."""
        new_route = route[:position] + [customer] + route[position:]
        # Check capacity
        total_demand = sum(demands[n] for n in new_route if n not in station_set)
        if total_demand > capacity:
            return False
        return check_route_feasibility(new_route)

    # Clarke-Wright savings
    def clarke_wright():
        # Start with each customer in its own route
        routes = [[c] for c in customers]
        route_index = {c: i for i, c in enumerate(customers)}

        # Compute savings
        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist_mat[depot, ci] + dist_mat[depot, cj] - dist_mat[ci, cj]
                savings.append((s, i, j))

        savings.sort(reverse=True)

        for s, i, j in savings:
            if time.time() > deadline - 5.0:
                break
            ci = customers[i]
            cj = customers[j]
            ri = route_index.get(ci)
            rj = route_index.get(cj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            # Check if ci is at the end of route_i and cj is at the start of route_j
            # or ci is at the start and cj is at the end
            merged = None
            if route_i[-1] == ci and route_j[0] == cj:
                merged = route_i + route_j
            elif route_i[0] == ci and route_j[-1] == cj:
                merged = route_i + list(reversed(route_j))
            elif route_i[-1] == cj and route_j[0] == ci:
                merged = route_j + route_i
            elif route_i[0] == cj and route_j[-1] == ci:
                merged = list(reversed(route_j)) + route_i

            if merged is None:
                continue

            # Check feasibility
            total_demand = sum(demands[n] for n in merged if n not in station_set)
            if total_demand > capacity:
                continue

            if check_route_feasibility(merged):
                # Merge routes
                routes[ri] = merged
                for c in route_j:
                    route_index[c] = ri
                routes[rj] = []
                for c in route_j:
                    del route_index[c]

        # Remove empty routes
        routes = [r for r in routes if r]
        return routes

    # Or-opt and 2-opt local search
    def local_search(routes, max_time=30.0):
        """Apply 2-opt and Or-opt moves."""
        improved = True
        best_routes = [list(r) for r in routes]
        best_dist = sum(route_distance(r) for r in best_routes)

        while improved and time.time() < deadline - 10.0:
            improved = False
            start_ls = time.time()

            # 2-opt
            for ri in range(len(best_routes)):
                if time.time() > deadline - 10.0:
                    break
                route = best_routes[ri]
                n = len(route)
                for i in range(n - 1):
                    if time.time() > deadline - 10.0:
                        break
                    for j in range(i + 1, n):
                        if time.time() > deadline - 10.0:
                            break
                        # Try reversing segment i+1..j
                        new_route = route[:i + 1] + list(reversed(route[i + 1:j + 1])) + route[j + 1:]
                        total_demand = sum(demands[n2] for n2 in new_route if n2 not in station_set)
                        if total_demand > capacity:
                            continue
                        if not check_route_feasibility(new_route):
                            continue
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        if new_dist < old_dist - 1e-9:
                            best_routes[ri] = new_route
                            improved = True

            # Or-opt (move single node to another position in same route or another route)
            for ri in range(len(best_routes)):
                if time.time() > deadline - 10.0:
                    break
                route = best_routes[ri]
                for i in range(len(route)):
                    if time.time() > deadline - 10.0:
                        break
                    node = route[i]
                    # Try removing and reinserting in same route
                    remaining = route[:i] + route[i + 1:]
                    for pos in range(len(remaining) + 1):
                        if time.time() > deadline - 10.0:
                            break
                        new_route = remaining[:pos] + [node] + remaining[pos:]
                        if new_route == route:
                            continue
                        total_demand = sum(demands[n2] for n2 in new_route if n2 not in station_set)
                        if total_demand > capacity:
                            continue
                        if not check_route_feasibility(new_route):
                            continue
                        new_dist = route_distance(new_route)
                        old_dist = route_distance(route)
                        if new_dist < old_dist - 1e-9:
                            best_routes[ri] = new_route
                            improved = True
                            break

                    # Try moving to another route
                    for rj in range(len(best_routes)):
                        if rj == ri or time.time() > deadline - 10.0:
                            continue
                        target = best_routes[rj]
                        for pos in range(len(target) + 1):
                            if time.time() > deadline - 10.0:
                                break
                            new_target = target[:pos] + [node] + target[pos:]
                            new_source = route[:i] + route[i + 1:]
                            total_demand_t = sum(demands[n2] for n2 in new_target if n2 not in station_set)
                            total_demand_s = sum(demands[n2] for n2 in new_source if n2 not in station_set)
                            if total_demand_t > capacity or total_demand_s > capacity:
                                continue
                            if not check_route_feasibility(new_target):
                                continue
                            if new_source and not check_route_feasibility(new_source):
                                continue
                            new_dist = route_distance(new_target) + (route_distance(new_source) if new_source else 0)
                            old_dist = route_distance(route) + route_distance(target)
                            if new_dist < old_dist - 1e-9:
                                best_routes[ri] = new_source
                                best_routes[rj] = new_target
                                improved = True
                                break
                        if improved:
                            break

        return best_routes

    # Generate initial solution using Clarke-Wright
    routes = clarke_wright()

    # Local search
    routes = local_search(routes)

    # Final check
    total_dist = sum(route_distance(r) for r in routes)

    return routes
# EVOLVE-BLOCK-END
