"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt, inf
import time


def plan(instance):
    start_wall = time.time()
    deadline = start_wall + 130.0  # safety margin before 150s kill

    coords = np.array(instance["coords"], dtype=float)
    n = len(coords)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))

    # Precompute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    def simulate_route(route):
        """Simulate a route, return (distance, valid)."""
        if not route:
            return 0.0, True
        t = 0.0
        bat = float(battery_cap)
        load = 0.0
        total_dist = 0.0
        prev = 0
        for node in route:
            d = dist[prev][node]
            t += d / speed
            bat -= d
            if bat < -1e-9:
                return total_dist + d, False
            total_dist += d
            if node in stations:
                # Recharge
                if bat < 0:
                    bat = 0.0
                charge_time = bat * recharge_rate
                t += charge_time
                bat = float(battery_cap)
            else:
                # Customer
                load += demands[node]
                if load > capacity + 1e-9:
                    return total_dist, False
                t = max(t, ready[node])
                if t > due[node] + 1e-9:
                    return total_dist, False
                t += service[node]
            prev = node
        # Return to depot
        d = dist[prev][0]
        t += d / speed
        bat -= d
        if bat < -1e-9:
            return total_dist + d, False
        total_dist += d
        if t > horizon + 1e-9:
            return total_dist, False
        return total_dist, True

    def total_distance(routes):
        return sum(simulate_route(r)[0] for r in routes)

    def is_valid(routes):
        served = set()
        for r in routes:
            for node in r:
                if node in served:
                    return False
                served.add(node)
            d, v = simulate_route(r)
            if not v:
                return False
        if served != set(customers):
            return False
        return True

    def savings_method():
        """Clark-Wright savings heuristic."""
        # Start with each customer in its own route
        routes = {c: [c] for c in customers}
        # Compute savings
        savings_list = []
        for i in customers:
            for j in customers:
                if i != j:
                    s = dist[0][i] + dist[0][j] - dist[i][j]
                    savings_list.append((s, i, j))
        savings_list.sort(reverse=True)

        route_of = {c: [c] for c in customers}
        # Map customer to route id (use first customer as id)
        cust_to_route = {c: c for c in customers}

        for s, i, j in savings_list:
            if s <= 0:
                continue
            ri = cust_to_route.get(i)
            rj = cust_to_route.get(j)
            if ri is None or rj is None or ri == rj:
                continue
            # Check if i is at end of its route and j is at start of its route
            route_i = route_of[ri]
            route_j = route_of[rj]
            if route_i[-1] == i and route_j[0] == j:
                # Merge: route_i + route_j
                new_route = route_i + route_j
                # Check feasibility
                d, valid = simulate_route(new_route)
                if valid:
                    for c in new_route:
                        cust_to_route[c] = ri
                    route_of[ri] = new_route
                    del route_of[rj]
            elif route_j[-1] == j and route_i[0] == i:
                new_route = route_j + route_i
                d, valid = simulate_route(new_route)
                if valid:
                    for c in new_route:
                        cust_to_route[c] = rj
                    route_of[rj] = new_route
                    del route_of[ri]

        return list(route_of.values())

    def nearest_neighbor():
        """Nearest neighbor heuristic."""
        remaining = set(customers)
        routes = []
        while remaining:
            route = []
            prev = 0
            t = 0.0
            bat = float(battery_cap)
            load = 0.0
            while True:
                best_node = None
                best_cost = inf
                for c in remaining:
                    d = dist[prev][c]
                    # Check if we can reach and serve
                    t2 = t + d / speed
                    bat2 = bat - d
                    if bat2 < -1e-9:
                        continue
                    t2 = max(t2, ready[c])
                    if t2 > due[c] + 1e-9:
                        continue
                    if load + demands[c] > capacity + 1e-9:
                        continue
                    # Check return feasibility roughly
                    d_ret = dist[c][0]
                    if t2 + service[c] + d_ret / speed > horizon + 1e-9:
                        continue
                    if bat2 - d_ret < -1e-9:
                        continue
                    cost = d
                    if cost < best_cost:
                        best_cost = cost
                        best_node = c
                if best_node is None:
                    break
                d = dist[prev][best_node]
                t += d / speed
                bat -= d
                t = max(t, ready[best_node])
                t += service[best_node]
                load += demands[best_node]
                route.append(best_node)
                remaining.discard(best_node)
                prev = best_node
            if route:
                routes.append(route)
        return routes

    def cluster_heuristic(k):
        """Simple k-means style clustering then TSP per cluster."""
        # Initialize centers randomly
        np.random.seed(42)
        centers = np.random.choice(customers, size=k, replace=False)
        center_coords = coords[centers]

        for _ in range(10):
            # Assign customers to nearest center
            dists = np.array([np.sqrt(((coords[c] - cc) ** 2).sum()) for c in customers])
            labels = np.argmin(dists, axis=1)

            # Update centers
            new_centers = []
            for i in range(k):
                members = [customers[j] for j in range(len(customers)) if labels[j] == i]
                if members:
                    new_centers.append(coords[members].mean(axis=0))
                else:
                    new_centers.append(center_coords[i])
            center_coords = np.array(new_centers)

        # Build routes per cluster
        routes = []
        for i in range(k):
            members = [customers[j] for j in range(len(customers)) if labels[j] == i]
            if not members:
                continue
            # Sort by angle from depot
            angles = np.arctan2(coords[members][:, 1] - coords[0][1],
                                coords[members][:, 0] - coords[0][0])
            order = np.argsort(angles)
            route = [members[i] for i in order]
            routes.append(route)
        return routes

    def local_search_2opt(routes):
        """Apply 2-opt and reinsertion moves."""
        improved = True
        while improved:
            improved = False
            for ridx, route in enumerate(routes):
                if len(route) < 2:
                    continue
                # 2-opt
                for i in range(len(route) - 1):
                    for j in range(i + 1, len(route)):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        d_old, _ = simulate_route(route)
                        d_new, valid = simulate_route(new_route)
                        if valid and d_new < d_old - 1e-9:
                            routes[ridx] = new_route
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
                # Or-opt: move single node
                for i in range(len(route)):
                    node = route[i]
                    rest = route[:i] + route[i+1:]
                    for j in range(len(rest) + 1):
                        new_route = rest[:j] + [node] + rest[j:]
                        if new_route == route:
                            continue
                        d_old, _ = simulate_route(route)
                        d_new, valid = simulate_route(new_route)
                        if valid and d_new < d_old - 1e-9:
                            routes[ridx] = new_route
                            improved = True
                            break
                    if improved:
                        break
            # Inter-route moves: move a customer from one route to another
            for i in range(len(routes)):
                for j in range(len(routes)):
                    if i == j or not routes[i]:
                        continue
                    for k in range(len(routes[i])):
                        node = routes[i][k]
                        remaining = routes[i][:k] + routes[i][k+1:]
                        for pos in range(len(routes[j]) + 1):
                            new_j = routes[j][:pos] + [node] + routes[j][pos:]
                            d_old_i, _ = simulate_route(routes[i])
                            d_old_j, _ = simulate_route(routes[j])
                            d_new_i, valid_i = simulate_route(remaining)
                            d_new_j, valid_j = simulate_route(new_j)
                            if valid_i and valid_j:
                                if d_new_i + d_new_j < d_old_i + d_old_j - 1e-9:
                                    routes[i] = remaining
                                    routes[j] = new_j
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            # Remove empty routes
            routes = [r for r in routes if r]
        return routes

    # Generate multiple initial solutions
    def generate_initial():
        solutions = []
        solutions.append(savings_method())
        solutions.append(nearest_neighbor())
        for k in [3, 4, 5]:
            solutions.append(cluster_heuristic(k))
        return solutions

    # LNS: Large Neighborhood Search
    def lns(initial_routes, max_iters=5000):
        best_routes = [list(r) for r in initial_routes]
        best_dist = sum(simulate_route(r)[0] for r in best_routes)
        current_routes = [list(r) for r in initial_routes]
        current_dist = best_dist

        all_customers_set = set(customers)

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Destroy: remove random subset of customers
            n_remove = max(1, min(5, len(all_customers_set)))
            to_remove = set(np.random.choice(list(all_customers_set), size=n_remove, replace=False))

            # Remove from current routes
            removed_customers = []
            new_current = []
            for route in current_routes:
                new_route = []
                for node in route:
                    if node in to_remove:
                        removed_customers.append(node)
                    else:
                        new_route.append(node)
                if new_route:
                    new_current.append(new_route)
            current_routes = new_current

            # Reconstruct: insert removed customers one by one at best position
            # Sort by due date for insertion order
            removed_customers.sort(key=lambda c: due[c])

            for c in removed_customers:
                best_pos = None
                best_cost = inf
                # Try inserting into each existing route
                for ridx, route in enumerate(current_routes):
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        d, valid = simulate_route(new_route)
                        if valid:
                            if d < best_cost:
                                best_cost = d
                                best_pos = (ridx, pos)
                # Try creating a new route
                d, valid = simulate_route([c])
                if valid and d < best_cost:
                    best_cost = d
                    best_pos = (-1, 0)

                if best_pos is not None:
                    ridx, pos = best_pos
                    if ridx == -1:
                        current_routes.append([c])
                    else:
                        current_routes[ridx] = current_routes[ridx][:pos] + [c] + current_routes[ridx][pos:]

            # Evaluate
            if is_valid(current_routes):
                curr_dist = sum(simulate_route(r)[0] for r in current_routes)
                if curr_dist < current_dist:
                    current_dist = curr_dist
                    current_routes = [list(r) for r in current_routes]
                    if curr_dist < best_dist:
                        best_dist = curr_dist
                        best_routes = [list(r) for r in current_routes]
                        # Local search on best
                        best_routes = local_search_2opt(best_routes)
                        best_dist = sum(simulate_route(r)[0] for r in best_routes)
            else:
                # Revert
                current_routes = [list(r) for r in best_routes]
                current_dist = best_dist

        return best_routes

    # Main: try multiple heuristics, pick best, then LNS
    best_overall = None
    best_overall_dist = inf

    for init_sol in generate_initial():
        if time.time() > deadline:
            break
        # Validate
        if not is_valid(init_sol):
            continue
        d = sum(simulate_route(r)[0] for r in init_sol)
        if d < best_overall_dist:
            best_overall_dist = d
            best_overall = [list(r) for r in init_sol]

    if best_overall is None:
        # Fallback: one customer per route
        return [[c] for c in customers]

    # Apply local search
    best_overall = local_search_2opt(best_overall)
    best_overall_dist = sum(simulate_route(r)[0] for r in best_overall)

    # Apply LNS
    if time.time() < deadline - 10:
        lns_result = lns(best_overall)
        if is_valid(lns_result):
            lns_dist = sum(simulate_route(r)[0] for r in lns_result)
            if lns_dist < best_overall_dist:
                best_overall = lns_result
                best_overall_dist = lns_dist

    return best_overall
# EVOLVE-BLOCK-END
