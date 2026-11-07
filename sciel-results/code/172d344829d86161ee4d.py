"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin

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

    # Precompute distance matrix using numpy
    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    # Simulate a route to check feasibility and get metrics
    # Route can contain customer ids and station ids
    def simulate_route(route):
        """
        Simulate a route (list of node ids - customers and/or stations).
        Returns (feasible, total_dist, end_time, end_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery

        cur_time = 0.0
        cur_battery = battery
        total_dist = 0.0
        cur_pos = 0

        for node in route:
            d = dist[cur_pos][node]
            cur_time += d / speed
            cur_battery -= d
            total_dist += d

            if cur_battery < -1e-9:
                return False, total_dist, cur_time, cur_battery

            if node in station_set:
                # Charging station: add recharge time and refill
                energy_needed = battery - cur_battery
                recharge_time = energy_needed * recharge_rate
                cur_time += recharge_time
                cur_battery = battery
            else:
                # Customer: check time window
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist, cur_time, cur_battery
                cur_time += service[node]

            cur_pos = node

        # Return to depot
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        total_dist += d

        if cur_battery < -1e-9:
            return False, total_dist, cur_time, cur_battery

        if cur_time > horizon + 1e-9:
            return False, total_dist, cur_time, cur_battery

        return True, total_dist, cur_time, cur_battery

    def route_total_dist(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i + 1]]
        d += dist[route[-1]][0]
        return d

    def route_demand(route):
        total = 0
        for node in route:
            if node not in station_set:
                total += demands[node]
        return total

    # Try to insert a charging station into a route at the best position
    # to make it feasible
    def try_insert_station(route):
        """Try to find a position to insert a station to make route feasible."""
        # First check if route is feasible without station
        feasible, _, _, _ = simulate_route(route)
        if feasible:
            return route

        # Try inserting each station at each position
        best_route = None
        best_dist = float('inf')

        for pos in range(len(route) + 1):
            for st in stations:
                new_route = route[:pos] + [st] + route[pos:]
                feasible, total_d, _, _ = simulate_route(new_route)
                if feasible:
                    if total_d < best_dist:
                        best_dist = total_d
                        best_route = new_route

        if best_route is not None:
            return best_route
        return None

    # Construction Heuristic: Nearest Neighbor with constraints
    def construct_routes():
        unvisited = set(customers)
        routes = []

        while unvisited:
            if time.time() > deadline:
                break
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery
            route_dem = 0
            total_dist = 0.0

            while unvisited:
                best_cust = None
                best_score = float('inf')

                for cust in unvisited:
                    d = dist[cur_pos][cust]
                    new_time = cur_time + d / speed
                    new_battery = cur_battery - d

                    if new_battery < -1e-9:
                        continue
                    if route_dem + demands[cust] > capacity:
                        continue

                    new_time = max(new_time, ready[cust])
                    if new_time > due[cust] + 1e-9:
                        continue

                    # Check return to depot feasibility
                    return_d = dist[cust][0]
                    if new_battery - return_d < -1e-9:
                        continue
                    if new_time + service[cust] + return_d / speed > horizon + 1e-9:
                        continue

                    # Score: prefer nearest, but also consider time window tightness
                    if d < best_score:
                        best_score = d
                        best_cust = cust

                if best_cust is None:
                    break

                # Commit to best_cust
                d = dist[cur_pos][best_cust]
                cur_time += d / speed
                cur_battery -= d
                total_dist += d
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                route_dem += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust

            if route:
                # Try to make feasible, possibly with station insertion
                feasible_route = try_insert_station(route)
                if feasible_route is not None:
                    routes.append(feasible_route)
                else:
                    # Rollback: put them back and try one-by-one
                    unvisited.update(route)
                    for cust in list(route):
                        if cust in unvisited:
                            one_route = [cust]
                            feasible_route = try_insert_station(one_route)
                            if feasible_route is not None:
                                routes.append(feasible_route)
                                unvisited.remove(cust)
            else:
                # No customer could be added to current route
                for cust in list(unvisited):
                    one_route = [cust]
                    feasible_route = try_insert_station(one_route)
                    if feasible_route is not None:
                        routes.append(feasible_route)
                        unvisited.remove(cust)
                        break

        # Handle any remaining unvisited
        if unvisited:
            for cust in unvisited:
                one_route = [cust]
                feasible_route = try_insert_station(one_route)
                if feasible_route is not None:
                    routes.append(feasible_route)
                else:
                    routes.append(one_route)

        return routes

    routes = construct_routes()

    # Fast feasibility check for a route (used in local search)
    def is_feasible(route):
        if not route:
            return True
        cur_time = 0.0
        cur_battery = battery
        cur_pos = 0
        for node in route:
            d = dist[cur_pos][node]
            cur_time += d / speed
            cur_battery -= d
            if cur_battery < -1e-9:
                return False
            if node in station_set:
                energy_needed = battery - cur_battery
                cur_time += energy_needed * recharge_rate
                cur_battery = battery
            else:
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False
                cur_time += service[node]
            cur_pos = node
        d = dist[cur_pos][0]
        cur_time += d / speed
        cur_battery -= d
        if cur_battery < -1e-9:
            return False
        if cur_time > horizon + 1e-9:
            return False
        return True

    # Fast distance calculation
    def fast_route_dist(route):
        if not route:
            return 0.0
        d = dist[0][route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i]][route[i + 1]]
        d += dist[route[-1]][0]
        return d

    # Improvement Phase: Inter-route moves with fast distance delta
    def try_improve(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n_routes = len(routes)
            for i in range(n_routes):
                if not routes[i]:
                    continue
                route_i = routes[i]
                for j in range(len(route_i)):
                    cust = route_i[j]
                    if cust in station_set:
                        continue  # Don't move stations in this pass

                    # Remove cust from route i
                    new_route_i = route_i[:j] + route_i[j+1:]
                    if not new_route_i:
                        feasible_i = True
                        dist_i_old = 0.0
                        dist_i_new = 0.0
                    else:
                        feasible_i = is_feasible(new_route_i)
                        if not feasible_i:
                            continue
                        dist_i_old = fast_route_dist(route_i)
                        dist_i_new = fast_route_dist(new_route_i)

                    # Try inserting into all other routes
                    for k in range(n_routes):
                        if k == i:
                            continue
                        route_k = routes[k]
                        # Check capacity
                        dem_k = route_demand(route_k)
                        if dem_k + demands[cust] > capacity:
                            continue

                        for pos in range(len(route_k) + 1):
                            new_route_k = route_k[:pos] + [cust] + route_k[pos:]
                            if not is_feasible(new_route_k):
                                continue

                            dist_k_old = fast_route_dist(route_k)
                            dist_k_new = fast_route_dist(new_route_k)

                            old_total = dist_i_old + dist_k_old
                            new_total = dist_i_new + dist_k_new

                            if new_total < old_total - 1e-6:
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

            # Cleanup empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes

        return routes

    routes = try_improve(routes)

    # Intra-route 2-opt with fast distance delta
    def try_2opt(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(routes)):
                route = routes[i]
                if len(route) < 3:
                    continue
                changed = False
                n = len(route)
                for a in range(n - 1):
                    for b in range(a + 1, n):
                        new_route = route[:a+1] + route[a+1:b+1][::-1] + route[b+1:]
                        if not is_feasible(new_route):
                            continue
                        old_d = fast_route_dist(route)
                        new_d = fast_route_dist(new_route)
                        if new_d < old_d - 1e-6:
                            routes[i] = new_route
                            improved = True
                            changed = True
                            break
                    if changed:
                        break
                if changed:
                    break
        return routes

    routes = try_2opt(routes)

    # Or-opt: move 1 or 2 consecutive nodes to a different position
    def try_oropt(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(routes)):
                route = routes[i]
                if len(route) < 3:
                    continue
                n = len(route)
                # Try moving 1 node
                for length in [1, 2]:
                    for start in range(n - length + 1):
                        segment = route[start:start+length]
                        if any(s in station_set for s in segment):
                            continue
                        remaining = route[:start] + route[start+length:]
                        if not remaining:
                            continue
                        if not is_feasible(remaining):
                            continue
                        for pos in range(len(remaining) + 1):
                            new_route = remaining[:pos] + segment + remaining[pos:]
                            if new_route == route:
                                continue
                            if not is_feasible(new_route):
                                continue
                            old_d = fast_route_dist(route)
                            new_d = fast_route_dist(new_route)
                            if new_d < old_d - 1e-6:
                                routes[i] = new_route
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        return routes

    routes = try_oropt(routes)

    # Final check: ensure all customers are present
    all_served = set()
    for r in routes:
        for node in r:
            if node not in station_set:
                all_served.add(node)
    if all_served != set(customers):
        missing = set(customers) - all_served
        for m in missing:
            one_route = [m]
            feasible_route = try_insert_station(one_route)
            if feasible_route is not None:
                routes.append(feasible_route)
            else:
                routes.append(one_route)

    return routes
# EVOLVE-BLOCK-END
