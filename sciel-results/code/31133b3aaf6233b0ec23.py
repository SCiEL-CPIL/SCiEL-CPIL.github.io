"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin before 150s kill

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

    n_nodes = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    customers = list(range(1, n_customers + 1))
    station_set = set(stations)

    def simulate_route(route):
        """
        Simulate a route (list of node ids - customers and/or stations).
        Returns (feasible, total_dist, end_time, end_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery_cap

        cur_time = 0.0
        cur_battery = battery_cap
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
                # Recharge
                charge_time = (battery_cap - cur_battery) * recharge_rate
                cur_time += charge_time
                cur_battery = battery_cap
                cur_pos = node
            else:
                # Customer
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

    def route_demand(route):
        return sum(demands[c] for c in route if c not in station_set)

    def construct_savings():
        """
        Clarke-Wright savings construction with battery and time window constraints.
        """
        routes = [[c] for c in customers]

        savings = []
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                ci = customers[i]
                cj = customers[j]
                s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
                savings.append((s, i, j))

        savings.sort(reverse=True)

        cust_to_route = {c: i for i, c in enumerate(customers)}

        for s_val, i, j in savings:
            if time.time() > deadline:
                break
            ci = customers[i]
            cj = customers[j]

            ri = cust_to_route.get(ci)
            rj = cust_to_route.get(cj)
            if ri is None or rj is None or ri == rj:
                continue

            route_i = routes[ri]
            route_j = routes[rj]

            if route_demand(route_i) + route_demand(route_j) > capacity:
                continue

            merged = None
            # Try merging
            for orient_i in [route_i, route_i[::-1]]:
                for orient_j in [route_j, route_j[::-1]]:
                    candidate = orient_i + orient_j
                    feasible, _, _, _ = simulate_route(candidate)
                    if feasible:
                        merged = candidate
                        break
                if merged:
                    break

            if merged is None:
                # Try with station insertion
                for orient_i in [route_i, route_i[::-1]]:
                    for orient_j in [route_j, route_j[::-1]]:
                        candidate = orient_i + orient_j
                        for st in stations:
                            for pos in range(len(candidate) + 1):
                                test_route = candidate[:pos] + [st] + candidate[pos:]
                                if route_demand(test_route) > capacity:
                                    continue
                                f, _, _, _ = simulate_route(test_route)
                                if f:
                                    merged = test_route
                                    break
                            if merged:
                                break
                        if merged:
                            break
                    if merged:
                        break

            if merged is not None:
                routes[ri] = merged
                routes[rj] = []
                for c in merged:
                    if c not in station_set:
                        cust_to_route[c] = ri

        routes = [r for r in routes if r]
        
        # Ensure all customers are present
        all_served = set()
        for r in routes:
            all_served.update(c for c in r if c not in station_set)
        missing = set(customers) - all_served
        for m in missing:
            routes.append([m])

        return routes

    def construct_nearest_neighbor():
        """Nearest neighbor construction with constraints."""
        unvisited = set(customers)
        routes = []

        while unvisited:
            if time.time() > deadline:
                break
            route = []
            cur_pos = 0
            cur_time = 0.0
            cur_battery = battery_cap
            route_dem = 0

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

                    return_d = dist[cust][0]
                    if new_battery - return_d < -1e-9:
                        continue
                    if new_time + service[cust] + return_d / speed > horizon + 1e-9:
                        continue

                    if d < best_score:
                        best_score = d
                        best_cust = cust

                if best_cust is None:
                    break

                d = dist[cur_pos][best_cust]
                cur_time += d / speed
                cur_battery -= d
                cur_time = max(cur_time, ready[best_cust])
                cur_time += service[best_cust]
                route_dem += demands[best_cust]
                route.append(best_cust)
                unvisited.remove(best_cust)
                cur_pos = best_cust

            if route:
                feasible, _, _, _ = simulate_route(route)
                if feasible:
                    routes.append(route)
                else:
                    # Try fixing with station
                    fixed_route = None
                    for st in stations:
                        for pos in range(len(route) + 1):
                            test = route[:pos] + [st] + route[pos:]
                            if simulate_route(test)[0]:
                                fixed_route = test
                                break
                        if fixed_route:
                            break
                    if fixed_route:
                        routes.append(fixed_route)
                    else:
                        for c in route:
                            routes.append([c])
                            unvisited.add(c)
            else:
                for cust in list(unvisited):
                    one_route = [cust]
                    if simulate_route(one_route)[0]:
                        routes.append(one_route)
                        unvisited.remove(cust)
                        break
                else:
                    for cust in list(unvisited):
                        found = False
                        for st in stations:
                            if simulate_route([st, cust])[0]:
                                routes.append([st, cust])
                                unvisited.remove(cust)
                                found = True
                                break
                            if simulate_route([cust, st])[0]:
                                routes.append([cust, st])
                                unvisited.remove(cust)
                                found = True
                                break
                        if found:
                            break

        if unvisited:
            for cust in list(unvisited):
                routes.append([cust])

        return routes

    # Try both constructions
    routes1 = construct_savings()
    dist1 = sum(simulate_route(r)[1] for r in routes1 if simulate_route(r)[0]) # Approx dist

    if time.time() < deadline - 10:
        routes2 = construct_nearest_neighbor()
        dist2 = sum(simulate_route(r)[1] for r in routes2 if simulate_route(r)[0])
        if dist2 < dist1:
            routes = routes2
        else:
            routes = routes1
    else:
        routes = routes1

    def two_opt_intra(routes, i):
        route = routes[i]
        if len(route) < 3:
            return False

        old_d, _, _, _ = simulate_route(route)
        if not old_d: # infeasible?
            return False
            
        best_improvement = 0
        best_a, best_b = -1, -1

        for a in range(len(route) - 1):
            for b in range(a + 1, len(route)):
                new_route = route[:a + 1] + route[a + 1:b + 1][::-1] + route[b + 1:]
                f, d, _, _ = simulate_route(new_route)
                if f:
                    imp = old_d - d
                    if imp > best_improvement:
                        best_improvement = imp
                        best_a, best_b = a, b

        if best_improvement > 1e-6:
            a, b = best_a, best_b
            routes[i] = route[:a + 1] + route[a + 1:b + 1][::-1] + route[b + 1:]
            return True
        return False

    def or_opt_move(routes, i, j, k):
        route = routes[i]
        if k - j > 3 or k > len(route) or j < 0:
            return False, routes

        segment = route[j:k]
        custs_in_seg = [c for c in segment if c not in station_set]
        if not custs_in_seg:
            return False, routes

        new_route_i = route[:j] + route[k:]
        f_i, d_i, _, _ = simulate_route(new_route_i)
        if not f_i:
            return False, routes

        old_total = d_i + 0 # d_i is dist of new_route_i
        
        # Actually, we need to compare total distance of the system
        # Let's just check if the new route is feasible and has lower distance than the segment was contributing
        # Simpler: Just try to insert segment elsewhere and check if total system distance decreases.
        
        # Try same route
        for pos in range(len(new_route_i) + 1):
            if pos >= j and pos <= k:
                continue
            new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
            if route_demand(new_route_i_test) > capacity:
                continue
            f, d, _, _ = simulate_route(new_route_i_test)
            if f and d < d_i - 1e-6: # d_i is dist of route WITHOUT segment? No.
                # d_i is dist of new_route_i (which is route WITHOUT segment).
                # Wait, if we remove segment, distance usually decreases.
                # We want to insert segment such that total distance is lower than original.
                # Original dist = dist(route)
                # New dist = dist(new_route_i_test)
                # We need dist(new_route_i_test) < dist(route)
                pass # Logic needs care. Let's stick to the previous implementation which worked.
        
        # Re-using the logic from the current program which was working
        old_route_dist, _, _, _ = simulate_route(route)
        
        # Try inserting into same route at different position
        for pos in range(len(new_route_i) + 1):
            if pos >= j and pos <= k:
                continue
            new_route_i_test = new_route_i[:pos] + segment + new_route_i[pos:]
            if route_demand(new_route_i_test) > capacity:
                continue
            f, d, _, _ = simulate_route(new_route_i_test)
            if f and d < old_route_dist - 1e-6:
                routes[i] = new_route_i_test
                return True, routes

        # Try inserting into other routes
        for k2 in range(len(routes)):
            if k2 == i:
                continue
            route_k = routes[k2]
            old_dist_k, _, _, _ = simulate_route(route_k)
            for pos in range(len(route_k) + 1):
                new_route_k = route_k[:pos] + segment + route_k[pos:]
                if route_demand(new_route_k) > capacity:
                    continue
                f, d, _, _ = simulate_route(new_route_k)
                if not f:
                    continue
                # Total distance change:
                # Old: dist(route_i) + dist(route_k)
                # New: dist(new_route_i) + dist(new_route_k)
                new_dist_total = d_i + d
                old_dist_total = old_route_dist + old_dist_k
                if new_dist_total < old_dist_total - 1e-6:
                    if not new_route_i:
                        routes[i] = []
                    else:
                        routes[i] = new_route_i
                    routes[k2] = new_route_k
                    return True, routes

        return False, routes

    def try_remove_station(routes, i):
        """Try to remove a station from route i if feasible."""
        route = routes[i]
        for j in range(len(route)):
            if route[j] in station_set:
                new_route = route[:j] + route[j+1:]
                if route_demand(new_route) <= capacity:
                    f, d, _, _ = simulate_route(new_route)
                    if f:
                        old_d, _, _, _ = simulate_route(route)
                        if d < old_d - 1e-6:
                            routes[i] = new_route
                            return True
        return False

    def try_swap(routes, i, j, ci, cj):
        """Try swapping customer ci in route i with customer cj in route j."""
        route_i = routes[i]
        route_j = routes[j]
        
        # Find indices
        idx_i = route_i.index(ci)
        idx_j = route_j.index(cj)
        
        new_route_i = route_i[:]
        new_route_j = route_j[:]
        
        new_route_i[idx_i] = cj
        new_route_j[idx_j] = ci
        
        if route_demand(new_route_i) > capacity or route_demand(new_route_j) > capacity:
            return False
        
        f_i, d_i, _, _ = simulate_route(new_route_i)
        f_j, d_j, _, _ = simulate_route(new_route_j)
        
        if f_i and f_j:
            old_d_i, _, _, _ = simulate_route(route_i)
            old_d_j, _, _, _ = simulate_route(route_j)
            
            if d_i + d_j < old_d_i + old_d_j - 1e-6:
                routes[i] = new_route_i
                routes[j] = new_route_j
                return True
        return False

    def try_improve(routes):
        improved = True
        while improved and time.time() < deadline:
            improved = False

            # 2-opt intra
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                if two_opt_intra(routes, i):
                    improved = True

            # Or-opt moves
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                route = routes[i]
                for j in range(len(route)):
                    for length in [1, 2, 3]:
                        k = j + length
                        if k > len(route):
                            continue
                        imp, routes = or_opt_move(routes, i, j, k)
                        if imp:
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

            # Remove stations
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                if try_remove_station(routes, i):
                    improved = True

            # Swaps (Inter-route)
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(routes)):
                    if time.time() > deadline:
                        break
                    route_i = routes[i]
                    route_j = routes[j]
                    # Only swap customers
                    for ci in route_i:
                        if ci in station_set: continue
                        for cj in route_j:
                            if cj in station_set: continue
                            if try_swap(routes, i, j, ci, cj):
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # Station insertion for infeasible routes
            for i in range(len(routes)):
                if time.time() > deadline:
                    break
                f, _, _, _ = simulate_route(routes[i])
                if not f:
                    # Try to fix
                    fixed = None
                    for st in stations:
                        for pos in range(len(routes[i]) + 1):
                            test = routes[i][:pos] + [st] + routes[i][pos:]
                            if simulate_route(test)[0]:
                                fixed = test
                                break
                        if fixed:
                            break
                    if fixed:
                        routes[i] = fixed
                        improved = True

            # Remove empty routes
            new_routes = [r for r in routes if r]
            if len(new_routes) != len(routes):
                routes = new_routes
                improved = True

        return routes

    routes = try_improve(routes)

    # Final validation and repair
    all_served = set()
    for r in routes:
        all_served.update(c for c in r if c not in station_set)

    missing = set(customers) - all_served
    for m in missing:
        inserted = False
        for i in range(len(routes)):
            route = routes[i]
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [m] + route[pos:]
                if route_demand(new_route) > capacity:
                    continue
                f, _, _, _ = simulate_route(new_route)
                if f:
                    routes[i] = new_route
                    inserted = True
                    break
            if inserted:
                break
        if not inserted:
            routes.append([m])

    # Final feasibility check
    for i in range(len(routes)):
        f, _, _, _ = simulate_route(routes[i])
        if not f:
            fixed = None
            for st in stations:
                for pos in range(len(routes[i]) + 1):
                    test = routes[i][:pos] + [st] + routes[i][pos:]
                    if simulate_route(test)[0]:
                        fixed = test
                        break
                if fixed:
                    break
            if fixed:
                routes[i] = fixed

    routes = [r for r in routes if r]
    return routes
# EVOLVE-BLOCK-END
