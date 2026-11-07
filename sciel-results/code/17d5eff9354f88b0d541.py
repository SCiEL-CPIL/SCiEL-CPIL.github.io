"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
            dist[i][j] = d
            dist[j][i] = d

    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        if not route:
            return True, 0.0

        # Check demand
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        # Check battery and time windows
        battery = battery_cap
        t = 0.0
        prev = 0  # depot

        for v in route:
            d = dist[prev][v]
            battery -= d
            t += d / speed
            if battery < -1e-9:
                return False, 0.0

            # Time window
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev][0]
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return False, 0.0
        if t > horizon + 1e-9:
            return False, 0.0

        # Total distance
        total_dist = sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))
        total_dist += dist[0][route[0]] + dist[route[-1]][0]
        return True, total_dist

    def check_route_with_stations(route, station_plan):
        """
        Check route feasibility with a specific station plan.
        route: list of customer node ids
        station_plan: list of (after_customer_index, station_node) pairs
        Returns (feasible, total_distance).
        """
        if not route:
            return True, 0.0

        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0

        battery = battery_cap
        t = 0.0
        prev = 0
        total_dist = 0.0

        station_iter = iter(enumerate(station_plan))
        current_station_idx = 0

        for i, v in enumerate(route):
            # Check if we need to visit a station before this customer
            while current_station_idx < len(station_plan) and station_plan[current_station_idx][0] == i:
                s_node = station_plan[current_station_idx][1]
                d = dist[prev][s_node]
                battery -= d
                t += d / speed
                if battery < -1e-9:
                    return False, 0.0
                # Recharge
                t += (battery_cap - battery) * recharge_rate
                battery = battery_cap
                total_dist += d
                prev = s_node
                current_station_idx += 1

            d = dist[prev][v]
            battery -= d
            t += d / speed
            if battery < -1e-9:
                return False, 0.0
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return False, 0.0
            t += service[v]
            total_dist += d
            prev = v

        # After last customer, check for trailing stations
        while current_station_idx < len(station_plan) and station_plan[current_station_idx][0] == len(route):
            s_node = station_plan[current_station_idx][1]
            d = dist[prev][s_node]
            battery -= d
            t += d / speed
            if battery < -1e-9:
                return False, 0.0
            t += (battery_cap - battery) * recharge_rate
            battery = battery_cap
            total_dist += d
            prev = s_node
            current_station_idx += 1

        # Return to depot
        d = dist[prev][0]
        battery -= d
        t += d / speed
        if battery < -1e-9:
            return False, 0.0
        if t > horizon + 1e-9:
            return False, 0.0
        total_dist += d

        return True, total_dist

    def find_station_plan(route):
        """
        Greedy: determine minimum stations needed for a route.
        Returns station_plan list or None if infeasible.
        """
        if not route:
            return []

        battery = battery_cap
        t = 0.0
        prev = 0
        station_plan = []

        for i, v in enumerate(route):
            d = dist[prev][v]
            if battery - d < -1e-9:
                # Need to go to a station first
                # Find closest station that we can reach
                best_station = None
                best_d = float('inf')
                for s in stations:
                    ds = dist[prev][s]
                    if battery - ds >= -1e-9 and ds < best_d:
                        best_d = ds
                        best_station = s
                if best_station is None:
                    return None
                # Go to station
                battery -= best_d
                t += best_d / speed
                t += (battery_cap - battery) * recharge_rate
                battery = battery_cap
                station_plan.append((i, best_station))
                prev = best_station
                d = dist[prev][v]

            battery -= d
            t += d / speed
            t = max(t, ready[v])
            if t > due[v] + 1e-9:
                return None
            t += service[v]
            prev = v

        # Return to depot
        d = dist[prev][0]
        if battery - d < -1e-9:
            # Need station before returning
            best_station = None
            best_d = float('inf')
            for s in stations:
                ds = dist[prev][s]
                if battery - ds >= -1e-9 and ds < best_d:
                    best_d = ds
                    best_station = s
            if best_station is None:
                return None
            battery -= best_d
            t += best_d / speed
            t += (battery_cap - battery) * recharge_rate
            battery = battery_cap
            station_plan.append((len(route), best_station))
            prev = best_station
            d = dist[prev][0]
            if battery - d < -1e-9:
                return None
            battery -= d
            t += d / speed
        else:
            battery -= d
            t += d / speed

        if t > horizon + 1e-9:
            return None

        return station_plan

    def route_total_dist(route):
        """Calculate total distance of a route (depot to first, between, last to depot)."""
        if not route:
            return 0.0
        total = dist[0][route[0]]
        for i in range(len(route) - 1):
            total += dist[route[i]][route[i + 1]]
        total += dist[route[-1]][0]
        return total

    def savings_merge(routes):
        """
        Use Clarke-Wright savings to merge routes.
        Returns new list of routes.
        """
        # Compute savings
        savings = {}
        customers_in_routes = {}
        for idx, route in enumerate(routes):
            for c in route:
                customers_in_routes[c] = idx

        # Build savings matrix for adjacent customers
        all_customers = set()
        for route in routes:
            all_customers.update(route)

        # For each pair of customers, compute savings if they're in different routes
        # and can be adjacent
        save_list = []
        for c1 in all_customers:
            for c2 in all_customers:
                if c1 >= c2:
                    continue
                r1 = customers_in_routes.get(c1)
                r2 = customers_in_routes.get(c2)
                if r1 is None or r2 is None or r1 == r2:
                    continue
                # Check if c1 is last in route r1 or c2 is first in route r2
                # or c2 is last in route r2 or c1 is first in route r1
                route1 = routes[r1]
                route2 = routes[r2]
                can_merge = False
                merge_type = None
                if route1[-1] == c1 and route2[0] == c2:
                    can_merge = True
                    merge_type = (r1, r2, 'end_start')
                elif route2[-1] == c2 and route1[0] == c1:
                    can_merge = True
                    merge_type = (r2, r1, 'end_start')
                if not can_merge:
                    continue
                s = dist[0][c1] + dist[0][c2] - dist[c1][c2]
                save_list.append((s, c1, c2, merge_type))

        save_list.sort(key=lambda x: -x[0])

        # Apply savings
        new_routes = [list(r) for r in routes]
        route_map = {}
        for idx, route in enumerate(new_routes):
            for c in route:
                route_map[c] = idx

        for s, c1, c2, (r1, r2, mtype) in save_list:
            if s <= 0:
                break
            # Check if still in separate routes
            idx1 = route_map.get(c1)
            idx2 = route_map.get(c2)
            if idx1 is None or idx2 is None or idx1 == idx2:
                continue
            if idx1 >= len(new_routes) or idx2 >= len(new_routes):
                continue
            route1 = new_routes[idx1]
            route2 = new_routes[idx2]
            if not route1 or not route2:
                continue

            # Verify merge is still valid
            if route1[-1] != c1 or route2[0] != c2:
                continue

            # Check capacity
            if sum(demands[v] for v in route1) + sum(demands[v] for v in route2) > capacity:
                continue

            # Check feasibility of merged route
            merged = route1 + route2
            sp = find_station_plan(merged)
            if sp is None:
                continue

            # Merge
            new_routes[idx1] = merged
            new_routes[idx2] = []
            for c in merged:
                route_map[c] = idx1

        # Remove empty routes
        return [r for r in new_routes if r]

    def two_opt(route):
        """Apply 2-opt to a route, return improved route and distance."""
        best_route = list(route)
        best_dist = route_total_dist(best_route)
        improved = True
        while improved:
            improved = False
            for i in range(len(best_route) - 1):
                for j in range(i + 2, len(best_route)):
                    if i == 0 and j == len(best_route) - 1:
                        continue
                    new_route = best_route[:i + 1] + best_route[i + 1:j + 1][::-1] + best_route[j + 1:]
                    # Check feasibility
                    sp = find_station_plan(new_route)
                    if sp is None:
                        continue
                    nd = route_total_dist(new_route)
                    if nd < best_dist - 1e-9:
                        best_route = new_route
                        best_dist = nd
                        improved = True
        return best_route, best_dist

    def or_opt(route):
        """Move a single customer or small segment to a better position."""
        best_route = list(route)
        best_dist = route_total_dist(best_route)
        n = len(best_route)
        for i in range(n):
            c = best_route[i]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                # Remove c from position i
                temp = best_route[:i] + best_route[i + 1:]
                # Insert at position j
                if j > i:
                    j -= 1
                new_route = temp[:j] + [c] + temp[j:]
                if new_route == best_route:
                    continue
                sp = find_station_plan(new_route)
                if sp is None:
                    continue
                nd = route_total_dist(new_route)
                if nd < best_dist - 1e-9:
                    best_route = new_route
                    best_dist = nd
        return best_route, best_dist

    def reassign_customer(routes, c):
        """Try to move customer c to a different route."""
        # Find current route
        cur_idx = None
        for idx, route in enumerate(routes):
            if c in route:
                cur_idx = idx
                break
        if cur_idx is None:
            return routes

        cur_route = routes[cur_idx]
        # Remove c
        cur_route.remove(c)

        best_routes = [list(r) for r in routes]
        best_total = sum(route_total_dist(r) for r in best_routes if r)

        # Try inserting c into each other route
        for idx, route in enumerate(routes):
            if idx == cur_idx or not route:
                continue
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                # Check capacity
                if sum(demands[v] for v in new_route) > capacity:
                    continue
                sp = find_station_plan(new_route)
                if sp is None:
                    continue
                # Compute new total
                new_total = 0
                for i, r in enumerate(routes):
                    if i == idx:
                        new_total += route_total_dist(new_route)
                    elif i == cur_idx:
                        new_total += route_total_dist(cur_route) if cur_route else 0
                    else:
                        new_total += route_total_dist(r)
                if new_total < best_total - 1e-9:
                    best_total = new_total
                    best_routes = [list(r) for r in routes]
                    best_routes[idx] = new_route
                    best_routes[cur_idx] = list(cur_route)

        # If no improvement, put c back
        if best_routes[cur_idx] != list(routes[cur_idx]) or any(
                best_routes[i] != list(routes[i]) for i in range(len(routes)) if i != cur_idx):
            return best_routes
        else:
            cur_route.append(c)  # put back
            # Actually we need to restore properly
            return [list(r) for r in routes]

    # --- Main algorithm ---
    # Step 1: Initial construction - one customer per route, sorted by due time
    customers = list(range(1, n_customers + 1))
    customers.sort(key=lambda v: due[v])
    routes = [[c] for c in customers]

    # Step 2: Savings-based merging
    max_iter = 50
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
        new_routes = savings_merge(routes)
        if len(new_routes) >= len(routes):
            break
        routes = new_routes

    # Step 3: Local search
    improved = True
    while improved:
        if time.time() > deadline - 5:
            break
        improved = False

        # 2-opt
        new_routes = []
        for route in routes:
            if len(route) < 2:
                new_routes.append(route)
                continue
            opt_route, _ = two_opt(route)
            if opt_route != route:
                improved = True
            new_routes.append(opt_route)
        routes = new_routes

        # Or-opt
        new_routes = []
        for route in routes:
            if len(route) < 2:
                new_routes.append(route)
                continue
            opt_route, _ = or_opt(route)
            if opt_route != route:
                improved = True
            new_routes.append(opt_route)
        routes = new_routes

        # Reassignment
        for c in customers:
            if time.time() > deadline - 5:
                break
            new_routes = reassign_customer(routes, c)
            if new_routes != routes:
                routes = new_routes
                improved = True

        # Another round of savings
        if time.time() > deadline - 5:
            break
        new_routes = savings_merge(routes)
        if len(new_routes) < len(routes):
            routes = new_routes
            improved = True

    # Step 4: Verify all customers are served
    served = set()
    for route in routes:
        for c in route:
            served.add(c)
    missing = set(range(1, n_customers + 1)) - served
    if missing:
        # Add missing customers as separate routes
        for c in missing:
            routes.append([c])

    # Step 5: Final verification and fallback
    final_routes = []
    for route in routes:
        if not route:
            continue
        sp = find_station_plan(route)
        if sp is not None:
            final_routes.append(route)
        else:
            # Split into individual customers
            for c in route:
                final_routes.append([c])

    # Ensure all customers are covered
    served = set()
    for route in final_routes:
        for c in route:
            served.add(c)
    missing = set(range(1, n_customers + 1)) - served
    for c in missing:
        final_routes.append([c])

    return final_routes
# EVOLVE-BLOCK-END
