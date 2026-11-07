import numpy as np
import time
import math
import random

def plan(instance):
    t0 = time.time()
    deadline = t0 + 140.0

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations_list = list(instance["stations"])
    stations = set(stations_list)
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    n_nodes = len(coords)
    dx = coords[:, 0]
    dy = coords[:, 1]
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        dist[i, :] = np.sqrt((dx[i] - dx)**2 + (dy[i] - dy)**2)

    customers = list(range(1, n_customers + 1))
    customer_set = set(customers)

    def evaluate_route(cust_seq):
        """
        Evaluates a sequence of customers.
        Returns (total_distance, feasible, full_route_with_stations)
        """
        if not cust_seq:
            return 0.0, True, []
        
        t = 0.0
        battery = battery_cap
        total_dist = 0.0
        current = 0
        full_route = []
        
        for c in cust_seq:
            d_direct = dist[current, c]
            if battery - d_direct >= -1e-9:
                # Reachable directly
                t += d_direct / speed
                battery -= d_direct
                total_dist += d_direct
                current = c
                full_route.append(c)
                
                if t < ready[c]: t = ready[c]
                if t > due[c] + 1e-9:
                    return (float('inf'), False, [])
                t += service[c]
            else:
                # Need station
                best_station = -1
                best_cost = float('inf')
                best_new_t = 0
                best_new_battery = 0
                best_new_dist = 0
                
                for s in stations_list:
                    d_to_s = dist[current, s]
                    if battery - d_to_s >= -1e-9:
                        t_s = t + d_to_s / speed
                        bat_s = battery - d_to_s
                        recharge_time = (battery_cap - bat_s) * recharge_rate
                        t_after_s = t_s + recharge_time
                        bat_after_s = battery_cap
                        
                        d_s_to_c = dist[s, c]
                        if bat_after_s - d_s_to_c >= -1e-9:
                            t_c = t_after_s + d_s_to_c / speed
                            bat_c = bat_after_s - d_s_to_c
                            
                            if t_c < ready[c]: t_c = ready[c]
                            if t_c > due[c] + 1e-9:
                                continue
                            t_c += service[c]
                            
                            cost = d_to_s + d_s_to_c
                            if cost < best_cost:
                                best_cost = cost
                                best_station = s
                                best_new_t = t_c
                                best_new_battery = bat_c
                                best_new_dist = cost
                                
                if best_station == -1:
                    return (float('inf'), False, [])
                
                t = best_new_t
                battery = best_new_battery
                total_dist += best_new_dist
                current = c
                full_route.append(best_station)
                full_route.append(c)
        
        # Return to depot
        d_back = dist[current, 0]
        if battery - d_back >= -1e-9:
            t += d_back / speed
            battery -= d_back
            total_dist += d_back
            if t > horizon + 1e-9:
                return (float('inf'), False, [])
            return (total_dist, True, full_route)
        else:
            best_station = -1
            best_cost = float('inf')
            best_new_t = 0
            best_new_battery = 0
            best_new_dist = 0
            
            for s in stations_list:
                d_to_s = dist[current, s]
                if battery - d_to_s >= -1e-9:
                    t_s = t + d_to_s / speed
                    bat_s = battery - d_to_s
                    recharge_time = (battery_cap - bat_s) * recharge_rate
                    t_after_s = t_s + recharge_time
                    bat_after_s = battery_cap
                    
                    d_s_to_0 = dist[s, 0]
                    if bat_after_s - d_s_to_0 >= -1e-9:
                        t_0 = t_after_s + d_s_to_0 / speed
                        bat_0 = bat_after_s - d_s_to_0
                        
                        if t_0 <= horizon + 1e-9:
                            cost = d_to_s + d_s_to_0
                            if cost < best_cost:
                                best_cost = cost
                                best_station = s
                                best_new_t = t_0
                                best_new_battery = bat_0
                                best_new_dist = cost
                                
            if best_station == -1:
                return (float('inf'), False, [])
            
            t = best_new_t
            battery = best_new_battery
            total_dist += best_new_dist
            full_route.append(best_station)
            return (total_dist, True, full_route)

    def route_demand(cust_seq):
        return sum(demands[v] for v in cust_seq)

    def total_distance(routes):
        total = 0.0
        for route in routes:
            sim = evaluate_route(route)
            if not sim[1]:
                return float('inf')
            total += sim[0]
        return total

    def best_insertion_cost(route, node):
        best_cost = float('inf')
        best_route = None
        if route_demand(route) + demands[node] > capacity:
            return (float('inf'), None)
        
        for pos in range(len(route) + 1):
            new_route = list(route[:pos]) + [node] + list(route[pos:])
            sim = evaluate_route(new_route)
            if sim[1]:
                if sim[0] < best_cost:
                    best_cost = sim[0]
                    best_route = new_route
        return (best_cost, best_route)

    def savings_based_construction():
        sorted_customers = sorted(customers, key=lambda v: due[v])
        unvisited = set(customers)
        built_routes = []
        
        for c in sorted_customers:
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
        return built_routes

    def two_opt(route, max_iter=50):
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
                    new_route = list(route[:i]) + list(reversed(route[i:j+1])) + list(route[j+1:])
                    if route_demand(new_route) > capacity:
                        continue
                    sim_new = evaluate_route(new_route)
                    if not sim_new[1]:
                        continue
                    sim_old = evaluate_route(route)
                    if not sim_old[1]:
                        continue
                    if sim_new[0] < sim_old[0] - 1e-6:
                        route = new_route
                        improved = True
            if time.time() > deadline - 5.0:
                break
        return route

    def or_opt(route, max_iter=30):
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
                    remaining = list(route[:i]) + list(route[i+length:])
                    if not remaining:
                        continue
                    for j in range(len(remaining) + 1):
                        new_route = list(remaining[:j]) + segment + list(remaining[j:])
                        if new_route == route:
                            continue
                        if route_demand(new_route) > capacity:
                            continue
                        sim_new = evaluate_route(new_route)
                        if not sim_new[1]:
                            continue
                        sim_old = evaluate_route(route)
                        if not sim_old[1]:
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
        for iteration in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            improved = False
            
            for i in range(len(routes)):
                new_route = two_opt(routes[i], 20)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True
            
            for i in range(len(routes)):
                new_route = or_opt(routes[i], 15)
                if new_route != routes[i]:
                    routes[i] = new_route
                    improved = True
            
            # Relocate
            for i in range(len(routes)):
                for pos_i, c in enumerate(routes[i]):
                    for j in range(len(routes)):
                        if i == j:
                            continue
                        if route_demand(routes[j]) + demands[c] > capacity:
                            continue
                        new_route_i = list(routes[i][:pos_i]) + list(routes[i][pos_i+1:])
                        cost, new_route_j = best_insertion_cost(routes[j], c)
                        if new_route_j is None:
                            continue
                        sim_i = evaluate_route(new_route_i) if new_route_i else (0.0, True, [])
                        sim_j = evaluate_route(new_route_j)
                        if not sim_i[1] or not sim_j[1]:
                            continue
                        old_dist = evaluate_route(routes[i])[0] + evaluate_route(routes[j])[0]
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
            
            routes = [r for r in routes if r]
            if not improved:
                break
        return routes

    routes = savings_based_construction()
    routes = local_search(routes, 50)
    best_routes = [list(r) for r in routes]
    best_dist = total_distance(best_routes)

    rng = random.Random(42)
    for lns_iter in range(200):
        if time.time() > deadline - 5.0:
            break
        n_remove = rng.randint(3, min(8, len(customers)))
        removed = rng.sample(customers, n_remove)
        removed_routes_idx = set()
        for i, route in enumerate(best_routes):
            for c in removed:
                if c in route:
                    removed_routes_idx.add(i)
                    break
        
        routes_copy = [list(r) for r in best_routes]
        for idx in removed_routes_idx:
            route = routes_copy[idx]
            for c in removed:
                if c in route:
                    route.remove(c)
        routes_copy = [r for r in routes_copy if r]
        
        for c in removed:
            best_route_idx = -1
            best_cost = float('inf')
            best_new_route = None
            for i, route in enumerate(routes_copy):
                cost, new_route = best_insertion_cost(route, c)
                if new_route is not None and cost < best_cost:
                    best_cost = cost
                    best_route_idx = i
                    best_new_route = new_route
            if best_route_idx >= 0:
                routes_copy[best_route_idx] = best_new_route
            else:
                routes_copy.append([c])
        
        routes_copy = local_search(routes_copy, 20)
        d = total_distance(routes_copy)
        if d < best_dist - 1e-6:
            best_routes = routes_copy
            best_dist = d

    # Final conversion to full routes
    final_routes = []
    for route in best_routes:
        sim = evaluate_route(route)
        if sim[1]:
            final_routes.append(sim[2])
        else:
            # Fallback
            final_routes.append(route)

    return final_routes
