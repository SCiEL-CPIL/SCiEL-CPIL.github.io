import numpy as np
import time
import heapq
import math
import random

def plan(instance):
    start_time = time.time()
    DEADLINE = 110.0

    # 1. Parse Input
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    customers = list(range(1, n_customers + 1))
    station_list = list(stations)

    # Precompute distance matrix
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist_list = dist.tolist()
    station_set = set(station_list)

    # 2. Feasibility & Cost Functions
    def is_feasible(route):
        if not route:
            return True, 0.0

        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        total_demand = 0.0
        dl = dist_list
        bm = battery_max
        cap = capacity
        hz = horizon
        st_set = station_set
        req = ready
        dued = due
        svc = service
        dem = demands
        rr = recharge_rate
        sp = speed

        for node in route:
            d = dl[curr_node][node]
            total_dist += d

            if curr_battery < d - 1e-9:
                return False, 0.0

            curr_battery -= d
            curr_time += d / sp

            if node in st_set:
                charge_time = (bm - curr_battery) * rr
                curr_time += charge_time
                curr_battery = bm
            else:
                total_demand += dem[node]
                if total_demand > cap + 1e-9:
                    return False, 0.0

                if curr_time < req[node]:
                    curr_time = req[node]
                if curr_time > dued[node] + 1e-9:
                    return False, 0.0

                curr_time += svc[node]

            curr_node = node

        d = dl[curr_node][0]
        total_dist += d

        if curr_battery < d - 1e-9:
            return False, 0.0

        curr_time += d / sp

        if curr_time > hz + 1e-9:
            return False, 0.0

        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        curr = 0
        total = 0.0
        dl = dist_list
        for node in route:
            total += dl[curr][node]
            curr = node
        total += dl[curr][0]
        return total

    def total_distance(routes):
        return sum(route_distance(r) for r in routes)

    def validate_solution(routes):
        visited = set()
        for route in routes:
            valid, _ = is_feasible(route)
            if not valid:
                return False
            for node in route:
                if node not in stations:
                    if node in visited:
                        return False
                    visited.add(node)
        return visited == set(customers)

    # 3. Battery-aware station insertion into a customer-only route
    def insert_stations(customer_route):
        """
        Given a sequence of customers, insert charging stations where needed
        to make the route feasible. Returns the full route with stations.
        """
        if not customer_route:
            return []
        
        route = []
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_demand = 0.0

        for c in customer_route:
            # Check if we can reach customer directly
            d_direct = dist[curr_node][c]
            if curr_battery >= d_direct - 1e-9:
                new_time = curr_time + d_direct / speed
                service_start = max(new_time, ready[c])
                if service_start <= due[c] + 1e-9:
                    route.append(c)
                    curr_battery -= d_direct
                    curr_time = service_start + service[c]
                    total_demand += demands[c]
                    curr_node = c
                    continue

            # Need to charge - find best station
            best_station = None
            best_total_dist = float('inf')
            for s in station_list:
                d1 = dist[curr_node][s]
                if curr_battery < d1 - 1e-9:
                    continue
                charge_time = (battery_max - (curr_battery - d1)) * recharge_rate
                new_time = curr_time + d1 / speed + charge_time
                d2 = dist[s][c]
                new_time2 = new_time + d2 / speed
                service_start = max(new_time2, ready[c])
                if service_start <= due[c] + 1e-9:
                    total_d = d1 + d2
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_station = s

            if best_station is None:
                return None  # Cannot make this route feasible

            # Go to station then to customer
            s = best_station
            d1 = dist[curr_node][s]
            curr_battery -= d1
            curr_time += d1 / speed
            charge_time = (battery_max - curr_battery) * recharge_rate
            curr_time += charge_time
            curr_battery = battery_max
            route.append(s)
            curr_node = s

            d2 = dist[curr_node][c]
            curr_battery -= d2
            curr_time += d2 / speed
            service_start = max(curr_time, ready[c])
            curr_time = service_start + service[c]
            total_demand += demands[c]
            curr_node = c

        # Check return to depot
        d_return = dist[curr_node][0]
        if curr_battery < d_return - 1e-9:
            # Need to charge before returning
            best_station = None
            best_total_dist = float('inf')
            for s in station_list:
                d1 = dist[curr_node][s]
                if curr_battery < d1 - 1e-9:
                    continue
                d2 = dist[s][0]
                total_d = d1 + d2
                if total_d < best_total_dist:
                    best_total_dist = total_d
                    best_station = s
            
            if best_station is None:
                return None
            
            s = best_station
            d1 = dist[curr_node][s]
            curr_battery -= d1
            curr_time += d1 / speed
            charge_time = (battery_max - curr_battery) * recharge_rate
            curr_time += charge_time
            curr_battery = battery_max
            route.append(s)
            curr_node = s
            
            d2 = dist[curr_node][0]
            curr_time += d2 / speed
            curr_battery -= d2
        
        else:
            curr_time += d_return / speed
            curr_battery -= d_return

        if curr_time > horizon + 1e-9:
            return None

        return route

    # 4. Savings-based construction (Clarke-Wright)
    def savings_construction():
        """
        Clarke-Wright savings heuristic with battery-aware feasibility.
        1. Start with individual routes for each customer.
        2. Compute savings for merging pairs.
        3. Merge pairs with positive savings while maintaining feasibility.
        """
        # Start with each customer as its own route
        routes = [[c] for c in customers]
        route_of_node = {c: i for i, c in enumerate(customers)}
        
        # Compute savings: savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
        # But we need to account for the full route with stations
        # For simplicity, use geometric savings and check feasibility after merging
        
        # Compute pairwise savings
        savings_list = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                c1 = routes[i][0]
                c2 = routes[j][0]
                s = dist_list[0][c1] + dist_list[0][c2] - dist_list[c1][c2]
                if s > 0:
                    savings_list.append((s, i, j))
        
        # Sort by savings descending
        savings_list.sort(reverse=True)
        
        # Merge routes with positive savings
        for s, i, j in savings_list:
            if time.time() - start_time > DEADLINE:
                break
            
            if i >= len(routes) or j >= len(routes):
                continue
            if route_of_node.get(routes[i][0]) != i or route_of_node.get(routes[j][0]) != j:
                continue
            
            # Try merging routes i and j
            # Try both orders: i then j, or j then i
            merged = None
            for order in [0, 1]:
                if order == 0:
                    combined_customers = routes[i] + routes[j]
                else:
                    combined_customers = routes[j] + routes[i]
                
                # Check capacity
                total_demand = sum(demands[c] for c in combined_customers)
                if total_demand > capacity + 1e-9:
                    continue
                
                # Try to insert stations and check feasibility
                full_route = insert_stations(combined_customers)
                if full_route is not None:
                    merged = full_route
                    break
            
            if merged is not None:
                # Remove routes j and i, add merged
                routes[i] = merged
                routes.pop(j)
                # Update route_of_node
                for c in customers:
                    if c in merged:
                        route_of_node[c] = i
        
        # Convert customer routes to full routes with stations
        final_routes = []
        for r in routes:
            full_route = insert_stations(r)
            if full_route is not None:
                final_routes.append(full_route)
            else:
                # Fallback: keep as is if possible
                final_routes.append(r)
        
        return final_routes

    # 5. Greedy insertion as fallback
    def greedy_insertion(sort_key="due"):
        """Greedy insertion heuristic."""
        routes = []
        if sort_key == "due":
            sorted_customers = sorted(customers, key=lambda v: due[v])
        elif sort_key == "angle":
            sorted_customers = sorted(customers, key=lambda v: np.arctan2(coords[v][1] - coords[0][1], coords[v][0] - coords[0][0]))
        else:
            sorted_customers = customers[:]

        for c in sorted_customers:
            if time.time() - start_time > DEADLINE:
                break

            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')

            for r_idx, route in enumerate(routes):
                current_demand = sum(demands[node] for node in route if node not in stations)
                if current_demand + demands[c] > capacity + 1e-9:
                    continue

                for p in range(len(route) + 1):
                    new_route = route[:p] + [c] + route[p:]
                    valid, actual_dist = is_feasible(new_route)
                    if valid and actual_dist < best_cost:
                        best_cost = actual_dist
                        best_route_idx = r_idx
                        best_pos = p

            if best_route_idx != -1:
                routes[best_route_idx].insert(best_pos, c)
            else:
                routes.append([c])

        return routes

    # 6. Local Search Operators
    def two_opt(route):
        """Improve route using 2-opt with delta evaluation."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return route

        n = len(best_route)
        if n < 3:
            return best_route

        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            for i in range(n - 1):
                if (time.time() - start_time) > DEADLINE:
                    return best_route
                for j in range(i + 1, n):
                    prev_node = best_route[i-1] if i > 0 else 0
                    next_node = best_route[j+1] if j+1 < n else 0
                    a = best_route[i]
                    b = best_route[j]
                    
                    old_dist = dist_list[prev_node][a] + dist_list[b][next_node]
                    new_dist = dist_list[prev_node][b] + dist_list[a][next_node]
                    
                    if new_dist < old_dist - 1e-9:
                        best_route[i:j+1] = best_route[i:j+1][::-1]
                        valid, total = is_feasible(best_route)
                        if valid and total < best_dist - 1e-9:
                            best_dist = total
                            improved = True
                            break
                if improved:
                    break
        return best_route

    def or_opt(route):
        """Improve route using OR-opt (move 1-2 nodes)."""
        if len(route) < 3:
            return route

        best_route = route[:]
        valid, best_dist = is_feasible(best_route)
        if not valid:
            return route

        n = len(best_route)
        if n < 3:
            return best_route

        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            for seg_len in [1, 2]:
                for i in range(n - seg_len + 1):
                    if (time.time() - start_time) > DEADLINE:
                        return best_route
                    segment = best_route[i:i+seg_len]
                    remaining = best_route[:i] + best_route[i+seg_len:]
                    for j in range(len(remaining) + 1):
                        if j == i or j == i + seg_len:
                            continue
                        new_route = remaining[:j] + segment + remaining[j:]
                        valid, new_dist = is_feasible(new_route)
                        if valid and new_dist < best_dist - 1e-9:
                            best_route = new_route
                            best_dist = new_dist
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return best_route

    def relocate_improve(routes):
        """Try moving customers between routes."""
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            for i in range(len(routes)):
                for j in range(len(routes[i])):
                    c = routes[i][j]
                    new_route_i = routes[i][:j] + routes[i][j+1:]
                    valid_i, dist_i = is_feasible(new_route_i)
                    if not valid_i:
                        continue
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            new_route_k = routes[k][:pos] + [c] + routes[k][pos:]
                            valid_k, dist_k = is_feasible(new_route_k)
                            if valid_k:
                                old_total = route_distance(routes[i]) + route_distance(routes[k])
                                new_total = dist_i + dist_k
                                if new_total < old_total - 1e-9:
                                    routes[i] = new_route_i
                                    routes[k] = new_route_k
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    def merge_routes(routes):
        """Try merging two routes if feasible."""
        improved = True
        while improved and (time.time() - start_time) < DEADLINE:
            improved = False
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    for order in [0, 1]:
                        if order == 0:
                            combined = routes[i] + routes[j]
                        else:
                            combined = routes[j] + routes[i]
                        valid, total_dist = is_feasible(combined)
                        if valid:
                            old_total = route_distance(routes[i]) + route_distance(routes[j])
                            if total_dist < old_total - 1e-9:
                                new_routes = routes[:i] + [combined] + routes[i+1:j] + routes[j+1:]
                                routes = new_routes
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
        return routes

    # 7. Main Plan
    initial_solutions = []

    # 1. Savings-based construction (primary)
    r_savings = savings_construction()
    if validate_solution(r_savings):
        initial_solutions.append(r_savings)

    # 2. Greedy by due date
    r1 = greedy_insertion("due")
    if validate_solution(r1):
        initial_solutions.append(r1)

    # 3. Greedy by angle
    r2 = greedy_insertion("angle")
    if validate_solution(r2):
        initial_solutions.append(r2)

    if not initial_solutions:
        return greedy_insertion("due")

    # Select best initial solution
    best_routes = min(initial_solutions, key=total_distance)

    # Local Search - multiple passes
    for pass_num in range(4):
        if (time.time() - start_time) > DEADLINE:
            break
        for r_idx in range(len(best_routes)):
            if (time.time() - start_time) > DEADLINE:
                break
            best_routes[r_idx] = two_opt(best_routes[r_idx])
        for r_idx in range(len(best_routes)):
            if (time.time() - start_time) > DEADLINE:
                break
            best_routes[r_idx] = or_opt(best_routes[r_idx])
        
        if (time.time() - start_time) < DEADLINE:
            best_routes = relocate_improve(best_routes)

        if (time.time() - start_time) < DEADLINE:
            best_routes = merge_routes(best_routes)

    # Final 2-opt pass
    for r_idx in range(len(best_routes)):
        if (time.time() - start_time) > DEADLINE:
            break
        best_routes[r_idx] = two_opt(best_routes[r_idx])

    # Final validation
    if not validate_solution(best_routes):
        return greedy_insertion("due")

    return best_routes

# EVOLVE-BLOCK-END
