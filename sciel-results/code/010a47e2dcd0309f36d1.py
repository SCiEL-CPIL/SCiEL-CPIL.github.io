"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin

    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix using NumPy for speed
    n_nodes = len(coords)
    # coords is (N, 2)
    diff = coords[:, np.newaxis] - coords[np.newaxis, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Ensure exact zeros on diagonal if needed, though sqrt(0) is 0
    np.fill_diagonal(dist, 0.0)

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """
        Simulate a route starting from depot.
        Returns (total_dist, feasible, end_time).
        """
        if not route:
            return 0.0, True, 0.0

        t = 0.0
        b = battery
        total_d = 0.0
        prev = 0

        for node in route:
            d = dist[prev][node]
            t += d / speed
            b -= d
            total_d += d

            if b < -1e-9:
                return total_d, False, t

            if node in station_set:
                # Recharge time
                if b < battery:
                    t += (battery - b) * recharge_rate
                b = battery
            else:
                # Customer
                if t < ready[node]:
                    t = ready[node]
                if t > due[node] + 1e-9:
                    return total_d, False, t
                t += service[node]

            prev = node

        # Return to depot
        d = dist[prev][0]
        t += d / speed
        b -= d
        total_d += d

        if b < -1e-9:
            return total_d, False, t
        if t > horizon + 1e-9:
            return total_d, False, t

        return total_d, True, t

    def total_demand(route):
        return sum(demands[v] for v in route if v not in station_set)

    def construct_greedy():
        unvisited = set(customers)
        routes = []

        # Sort customers by due time for tie-breaking
        cust_sorted = sorted(customers, key=lambda v: (due[v], dist[0][v]))

        while unvisited:
            # Start new route from depot
            route = []
            current_node = 0
            t = 0.0
            b = battery
            total_d = 0.0
            load = 0

            while unvisited:
                # Find nearest unvisited customer that can be added feasibly
                best_cust = None
                best_dist = float('inf')

                for c in cust_sorted:
                    if c not in unvisited:
                        continue
                    if load + demands[c] > capacity:
                        continue

                    # Check if we can reach c and serve it
                    d_to_c = dist[current_node][c]
                    new_b = b - d_to_c
                    if new_b < -1e-9:
                        continue

                    new_t = t + d_to_c / speed
                    if new_t < ready[c]:
                        new_t = ready[c]
                    if new_t > due[c] + 1e-9:
                        continue
                    new_t += service[c]

                    # Check if we can return to depot after serving c
                    d_back = dist[c][0]
                    final_b = new_b - d_back
                    final_t = new_t + d_back / speed
                    if final_b < -1e-9 or final_t > horizon + 1e-9:
                        continue

                    if d_to_c < best_dist:
                        best_dist = d_to_c
                        best_cust = c

                if best_cust is None:
                    break

                # Add customer to route
                d_to_c = dist[current_node][best_cust]
                t += d_to_c / speed
                b -= d_to_c
                total_d += d_to_c

                if t < ready[best_cust]:
                    t = ready[best_cust]
                t += service[best_cust]

                load += demands[best_cust]
                route.append(best_cust)
                current_node = best_cust
                unvisited.discard(best_cust)

            if route:
                # Check full route feasibility
                _, feasible, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Fallback: serve one by one
                    for c in route:
                        routes.append([c])
                    unvisited.update(route)

        return routes

    def two_opt_route(route):
        """Apply 2-opt to a route, maintaining feasibility."""
        if len(route) < 2:
            return route

        improved = True
        best_route = list(route)
        best_dist, _, _ = simulate_route(best_route)

        while improved and (time.time() < deadline - 5.0):
            improved = False
            for i in range(len(best_route) - 1):
                if time.time() >= deadline - 5.0:
                    break
                for j in range(i + 1, len(best_route)):
                    # Reverse segment [i+1..j]
                    new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                    nd, feasible, _ = simulate_route(new_route)
                    if feasible and nd < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = nd
                        improved = True
                        break
                if improved:
                    break

        return best_route

    def or_opt_insertion(routes):
        """
        Try to move a customer from one route to another to reduce total distance.
        """
        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            current_routes = [list(r) for r in routes]
            
            # Calculate current total distance
            curr_total_dist = 0
            for r in current_routes:
                d, _, _ = simulate_route(r)
                curr_total_dist += d

            for i in range(len(current_routes)):
                if time.time() >= deadline - 5.0:
                    return current_routes
                
                route_i = current_routes[i]
                if len(route_i) < 2:
                    continue
                    
                for k in range(len(route_i)):
                    cust = route_i[k]
                    # Remove cust from route_i
                    route_i_without = route_i[:k] + route_i[k+1:]
                    
                    # Check if route_i_without is still feasible
                    _, feas_i, _ = simulate_route(route_i_without) if route_i_without else (0, True, 0)
                    if not feas_i:
                        continue
                        
                    # Try inserting cust into other routes
                    for j in range(len(current_routes)):
                        if i == j:
                            continue
                        
                        route_j = current_routes[j]
                        
                        # Try all insertion positions in route_j
                        for pos in range(len(route_j) + 1):
                            new_route_j = route_j[:pos] + [cust] + route_j[pos:]
                            
                            # Check demand
                            if total_demand(new_route_j) > capacity:
                                continue
                            
                            # Check feasibility
                            d_j, feas_j, _ = simulate_route(new_route_j)
                            if not feas_j:
                                continue
                            
                            # Calculate new total distance
                            # Note: We need to re-evaluate route_i_without distance too
                            d_i, _, _ = simulate_route(route_i_without) if route_i_without else (0, True, 0)
                            
                            # Approximate improvement check:
                            # Old: dist(route_i) + dist(route_j)
                            # New: dist(route_i_without) + dist(new_route_j)
                            # We don't have old dists cached easily here without extra loops, 
                            # so we just check if the new configuration is valid and 
                            # potentially better by comparing against a global best found so far?
                            # For simplicity and speed, we accept if it reduces the specific 
                            # distance of the affected routes significantly or just accept 
                            # any valid move that doesn't increase total distance too much.
                            
                            # Let's just compute the delta for these two routes
                            old_d_i, _, _ = simulate_route(route_i)
                            old_d_j, _, _ = simulate_route(route_j)
                            old_sum = old_d_i + old_d_j
                            new_sum = d_i + d_j
                            
                            if new_sum < old_sum - 1e-9:
                                current_routes[i] = route_i_without
                                current_routes[j] = new_route_j
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
                
        return current_routes

    # Build initial solution
    routes = construct_greedy()

    # Apply 2-opt to each route
    for i in range(len(routes)):
        routes[i] = two_opt_route(routes[i])

    # Try to merge routes to reduce total distance (if capacity allows)
    def try_merge_routes(routes):
        merged = True
        while merged and (time.time() < deadline - 5.0):
            merged = False
            new_routes = []
            used = [False] * len(routes)

            for i in range(len(routes)):
                if used[i]:
                    continue
                for j in range(i + 1, len(routes)):
                    if used[j]:
                        continue
                    # Try concatenating routes[i] + routes[j]
                    combined = routes[i] + routes[j]
                    if total_demand(combined) > capacity:
                        continue
                    _, feasible, _ = simulate_route(combined)
                    if feasible:
                        new_routes.append(combined)
                        used[i] = True
                        used[j] = True
                        merged = True
                        break

            if merged:
                for k in range(len(routes)):
                    if not used[k]:
                        new_routes.append(routes[k])
                routes = new_routes

        return routes

    routes = try_merge_routes(routes)

    # Final 2-opt pass on merged routes
    for i in range(len(routes)):
        if time.time() < deadline - 5.0:
            routes[i] = two_opt_route(routes[i])

    # Apply Or-opt insertion to move customers between routes
    routes = or_opt_insertion(routes)
    
    # Final 2-opt pass after relocations
    for i in range(len(routes)):
        if time.time() < deadline - 5.0:
            routes[i] = two_opt_route(routes[i])

    # Verify all customers are covered
    all_served = set()
    for r in routes:
        for v in r:
            if v not in station_set:
                all_served.add(v)

    if all_served != set(customers):
        # Fallback: one customer per route
        routes = [[v] for v in customers]

    # Final feasibility check
    for r in routes:
        _, feasible, _ = simulate_route(r)
        if not feasible:
            # Fallback
            routes = [[v] for v in customers]
            break

    return routes
# EVOLVE-BLOCK-END
