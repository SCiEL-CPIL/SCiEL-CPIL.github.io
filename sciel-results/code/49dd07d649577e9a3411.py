"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations

def plan(instance):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # Extract data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    # Precompute distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))

    # Check if a route (list of node ids) is feasible
    def check_route(route):
        """Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        # Check capacity
        total_demand = sum(demands[v] for v in route if v <= n_customers)
        if total_demand > capacity:
            return False, 0.0

        # Simulate
        time_t = 0.0
        battery = battery_cap
        prev = 0
        total_dist = 0.0

        for v in route:
            d = dist[prev, v]
            time_t += d / speed
            battery -= d

            if battery < -1e-9:
                return False, 0.0

            if v <= n_customers:
                # Customer: wait if needed
                if time_t < ready[v]:
                    time_t = ready[v]
                if time_t > due[v] + 1e-9:
                    return False, 0.0
                time_t += service[v]
            else:
                # Station: recharge
                energy = battery_cap - battery
                if energy > 0:
                    time_t += energy * recharge_rate
                battery = battery_cap

            total_dist += d
            prev = v

        # Return to depot
        d = dist[prev, 0]
        time_t += d / speed
        battery -= d

        if battery < -1e-9:
            return False, 0.0
        if time_t > horizon + 1e-9:
            return False, 0.0

        total_dist += d
        return True, total_dist

    # Greedy construction: sort by due time, insert into best route
    def greedy_construct():
        """Build routes greedily."""
        # Sort customers by due time
        sorted_customers = sorted(customers, key=lambda v: due[v])

        routes = []  # list of lists of node ids

        for c in sorted_customers:
            best_route_idx = -1
            best_pos = -1
            best_dist = float('inf')

            for i, route in enumerate(routes):
                # Try inserting at each position
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    feasible, d = check_route(new_route)
                    if feasible and d < best_dist:
                        best_dist = d
                        best_route_idx = i
                        best_pos = pos

            if best_route_idx == -1:
                # Start a new route
                routes.append([c])
            else:
                routes[best_route_idx].insert(best_pos, c)

        return routes

    # Local search: 2-opt and relocation
    def local_search(routes, max_iters=500):
        """Improve routes using 2-opt and relocation."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1

            if time.time() - start_time > DEADLINE - 10:
                break

            # 2-opt within each route
            for i, route in enumerate(routes):
                n = len(route)
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        # Reverse segment a..b
                        new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                        feasible, d_new = check_route(new_route)
                        _, d_old = check_route(route)
                        if feasible and d_new < d_old - 1e-9:
                            routes[i] = new_route
                            improved = True

            # Relocation between routes
            for i, route_i in enumerate(routes):
                for j, route_j in enumerate(routes):
                    if i == j:
                        continue
                    for k in range(len(route_i)):
                        c = route_i[k]
                        new_route_i = route_i[:k] + route_i[k+1:]
                        # Try inserting c into route_j at each position
                        for pos in range(len(route_j) + 1):
                            new_route_j = route_j[:pos] + [c] + route_j[pos:]
                            feasible_i, d_i = check_route(new_route_i)
                            feasible_j, d_j = check_route(new_route_j)
                            if feasible_i and feasible_j:
                                old_d = 0
                                for r in routes:
                                    _, dd = check_route(r)
                                    old_d += dd
                                new_d = d_i + d_j
                                # Adjust for other routes
                                for m in range(len(routes)):
                                    if m != i and m != j:
                                        _, dd = check_route(routes[m])
                                        new_d += dd
                                if new_d < old_d - 1e-9:
                                    routes[i] = new_route_i
                                    routes[j] = new_route_j
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break

        return routes

    # Initial solution
    routes = greedy_construct()

    # Local search
    routes = local_search(routes)

    # Verify all customers are served exactly once
    served = set()
    for route in routes:
        for v in route:
            if v in served:
                # Invalid
                return [[v] for v in sorted(customers, key=lambda x: due[x])]
            served.add(v)

    if served != set(customers):
        # Invalid
        return [[v] for v in sorted(customers, key=lambda x: due[x])]

    return routes

# EVOLVE-BLOCK-END
