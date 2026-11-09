"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _build_dist_matrix(coords):
    n = len(coords)
    xs = np.array([c[0] for c in coords])
    ys = np.array([c[1] for c in coords])
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist


def _fast_check_route(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon):
    """Fast feasibility check with early exits. Returns (feasible, total_distance)."""
    if not route:
        return True, 0.0
    
    total_demand = 0
    for v in route:
        if v <= n_customers:
            total_demand += demands[v]
    if total_demand > capacity:
        return False, 0.0
    
    battery = battery_max
    time_t = 0.0
    total_dist = 0.0
    prev = 0
    
    for v in route:
        d = dist[prev][v]
        battery -= d
        total_dist += d
        time_t += d / speed
        
        if battery < -1e-9:
            return False, 0.0
        
        if v <= n_customers:
            if time_t < ready[v]:
                time_t = ready[v]
            if time_t > due[v] + 1e-9:
                return False, 0.0
            time_t += service[v]
        else:
            energy_needed = battery_max - battery
            time_t += energy_needed * recharge_rate
            battery = battery_max
        
        prev = v
    
    d = dist[prev][0]
    battery -= d
    total_dist += d
    time_t += d / speed
    
    if battery < -1e-9 or time_t > horizon + 1e-9:
        return False, 0.0
    
    return True, total_dist


def _route_distance(route, dist):
    """Compute total distance of a route including depot legs."""
    if not route:
        return 0.0
    total = dist[0][route[0]]
    for i in range(len(route) - 1):
        total += dist[route[i]][route[i+1]]
    total += dist[route[-1]][0]
    return total


def _construct_initial_routes(instance, dist):
    """Construct initial routes using a savings-based heuristic."""
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    routes = [[i] for i in range(1, n_customers + 1)]
    
    savings = []
    for i in range(n_customers):
        for j in range(i + 1, n_customers):
            ci = i + 1
            cj = j + 1
            s = dist[0][ci] + dist[0][cj] - dist[ci][cj]
            savings.append((s, ci, cj))
    
    savings.sort(reverse=True)
    
    route_of = {}
    for idx, r in enumerate(routes):
        for c in r:
            route_of[c] = idx
    
    route_demand = [0] * len(routes)
    for idx, r in enumerate(routes):
        for c in r:
            if c <= n_customers:
                route_demand[idx] += demands[c]
    
    for s, ci, cj in savings:
        ri = route_of.get(ci, -1)
        rj = route_of.get(cj, -1)
        if ri == -1 or rj == -1 or ri == rj:
            continue
        
        if route_demand[ri] + route_demand[rj] > capacity:
            continue
        
        best_new_route = None
        best_dist = float('inf')
        
        for order in [[routes[ri], routes[rj]], [routes[rj], routes[ri]]]:
            merged = order[0] + order[1]
            feasible, total_dist = _fast_check_route(merged, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
            if feasible and total_dist < best_dist:
                best_dist = total_dist
                best_new_route = merged
        
        if best_new_route is not None:
            routes[ri] = best_new_route
            routes[rj] = []
            route_demand[ri] += route_demand[rj]
            route_demand[rj] = 0
            for c in best_new_route:
                route_of[c] = ri
    
    routes = [r for r in routes if r]
    
    all_served = set()
    for r in routes:
        all_served.update(r)
    
    for c in range(1, n_customers + 1):
        if c not in all_served:
            inserted = False
            for idx, r in enumerate(routes):
                current_demand = sum(demands[v] for v in r if v <= n_customers)
                if current_demand + demands[c] > capacity:
                    continue
                best_pos = -1
                best_cost = float('inf')
                for pos in range(len(r) + 1):
                    new_route = r[:pos] + [c] + r[pos:]
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        if pos == 0:
                            inc = dist[0][c] + dist[c][r[0]] - dist[0][r[0]]
                        elif pos == len(r):
                            inc = dist[r[-1]][c] + dist[c][0] - dist[r[-1]][0]
                        else:
                            inc = dist[r[pos-1]][c] + dist[c][r[pos]] - dist[r[pos-1]][r[pos]]
                        if inc < best_cost:
                            best_cost = inc
                            best_pos = pos
                if best_pos >= 0:
                    routes[idx] = r[:best_pos] + [c] + r[best_pos:]
                    inserted = True
                    break
            if not inserted:
                routes.append([c])
    
    return routes


def _two_opt_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply 2-opt improvements to a route with distance-based pruning."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 20
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                prev_i = route[i-1] if i > 0 else 0
                next_j = route[j+1] if j < n - 1 else 0
                
                d_old = dist[prev_i][route[i]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _or_opt_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply OR-opt (relocate 1-2 consecutive nodes) improvements."""
    if len(route) < 3:
        return route
    
    n = len(route)
    improved = True
    max_passes = 10
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for seg_len in [1, 2]:
            for i in range(n - seg_len + 1):
                if time.time() > deadline:
                    return route
                segment = route[i:i+seg_len]
                rest = route[:i] + route[i+seg_len:]
                
                for pos in range(len(rest) + 1):
                    if time.time() > deadline:
                        return route
                    if pos == i or (pos > i and pos <= i + seg_len):
                        continue
                    
                    new_route = rest[:pos] + segment + rest[pos:]
                    
                    if len(new_route) == 0:
                        continue
                    
                    if pos == 0:
                        if len(rest) > 0:
                            d_old = dist[0][rest[0]] + dist[rest[-1]][0]
                            d_new = dist[0][segment[0]] + dist[segment[-1]][rest[0]] + dist[rest[-1]][0]
                        else:
                            d_old = 0
                            d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                    elif pos == len(rest):
                        if len(rest) > 0:
                            d_old = dist[rest[-1]][0]
                            d_new = dist[rest[-1]][segment[0]] + dist[segment[-1]][0]
                        else:
                            d_old = 0
                            d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                    else:
                        if len(rest) > 0:
                            d_old = dist[rest[pos-1]][rest[pos]]
                            d_new = dist[rest[pos-1]][segment[0]] + dist[segment[-1]][rest[pos]]
                        else:
                            d_old = 0
                            d_new = dist[0][segment[0]] + dist[segment[-1]][0]
                    
                    if d_new < d_old - 1e-9:
                        feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                        if feasible:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
    
    return route


def _swap_improve(route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Apply swap improvements to a route."""
    if len(route) < 2:
        return route
    
    n = len(route)
    improved = True
    max_passes = 8
    passes = 0
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            if time.time() > deadline:
                return route
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return route
                if route[i] > n_customers or route[j] > n_customers:
                    continue
                
                new_route = route[:]
                new_route[i], new_route[j] = new_route[j], new_route[i]
                
                if i == 0:
                    prev_i = 0
                else:
                    prev_i = route[i-1]
                if i == n-1:
                    next_i = 0
                else:
                    next_i = route[i+1]
                if j == 0:
                    prev_j = 0
                else:
                    prev_j = route[j-1]
                if j == n-1:
                    next_j = 0
                else:
                    next_j = route[j+1]
                
                d_old = dist[prev_i][route[i]] + dist[route[i]][next_i] + dist[prev_j][route[j]] + dist[route[j]][next_j]
                d_new = dist[prev_i][route[j]] + dist[route[j]][next_i] + dist[prev_j][route[i]] + dist[route[i]][next_j]
                
                if d_new < d_old - 1e-9:
                    feasible, _ = _fast_check_route(new_route, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible:
                        route = new_route
                        improved = True
                        break
            if improved:
                break
    
    return route


def _inter_route_move(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline):
    """Try to move a customer from one route to another to reduce total distance."""
    n_routes = len(routes)
    if n_routes < 2:
        return routes
    
    total_dist_before = sum(_route_distance(r, dist) for r in routes)
    
    for i in range(n_routes):
        if time.time() > deadline:
            return routes
        for j in range(n_routes):
            if i == j:
                continue
            if time.time() > deadline:
                return routes
            
            for k in range(len(routes[i])):
                if time.time() > deadline:
                    return routes
                if routes[i][k] > n_customers:
                    continue
                
                customer = routes[i][k]
                new_route_i = routes[i][:k] + routes[i][k+1:]
                
                if not new_route_i:
                    continue
                
                # Check if source route is still feasible
                feasible_src, _ = _fast_check_route(new_route_i, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                if not feasible_src:
                    continue
                
                # Try inserting customer into route j
                best_pos = -1
                best_cost = float('inf')
                for pos in range(len(routes[j]) + 1):
                    new_route_j = routes[j][:pos] + [customer] + routes[j][pos:]
                    feasible_dst, _ = _fast_check_route(new_route_j, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
                    if feasible_dst:
                        if pos == 0:
                            if len(routes[j]) > 0:
                                inc = dist[0][customer] + dist[customer][routes[j][0]] - dist[0][routes[j][0]]
                            else:
                                inc = dist[0][customer] + dist[customer][0]
                        elif pos == len(routes[j]):
                            inc = dist[routes[j][-1]][customer] + dist[customer][0] - dist[routes[j][-1]][0]
                        else:
                            inc = dist[routes[j][pos-1]][customer] + dist[customer][routes[j][pos]] - dist[routes[j][pos-1]][routes[j][pos]]
                        
                        # Subtract the savings from removing from route i
                        if k == 0:
                            if len(new_route_i) > 0:
                                saving = dist[0][customer] + dist[customer][new_route_i[0]] - dist[0][new_route_i[0]]
                            else:
                                saving = dist[0][customer] + dist[customer][0]
                        elif k == len(routes[i]) - 1:
                            saving = dist[new_route_i[-1]][customer] + dist[customer][0] - dist[new_route_i[-1]][0]
                        else:
                            saving = dist[routes[i][k-1]][customer] + dist[customer][routes[i][k+1]] - dist[routes[i][k-1]][routes[i][k+1]]
                        
                        net_cost = inc - saving
                        if net_cost < best_cost:
                            best_cost = net_cost
                            best_pos = pos
                
                if best_pos >= 0 and best_cost < -1e-9:
                    new_routes = [r[:] for r in routes]
                    new_routes[i] = new_route_i
                    new_routes[j] = routes[j][:best_pos] + [customer] + routes[j][best_pos:]
                    
                    total_dist_after = sum(_route_distance(r, dist) for r in new_routes if r)
                    if total_dist_after < total_dist_before - 1e-9:
                        routes = new_routes
                        total_dist_before = total_dist_after
                        return routes
    
    return routes


def _solve_instance(instance, time_limit=35):
    """Solve a single instance with time limit."""
    start_time = time.time()
    deadline = start_time + time_limit
    
    coords = instance["coords"]
    dist = _build_dist_matrix(coords)
    
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    capacity = instance["capacity"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    routes = _construct_initial_routes(instance, dist)
    
    # Phase 1: Intensive local search on each route
    while time.time() < deadline - 5:
        for i in range(len(routes)):
            if time.time() > deadline - 5:
                break
            routes[i] = _two_opt_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 5)
            if time.time() > deadline - 5:
                break
            routes[i] = _or_opt_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 5)
            if time.time() > deadline - 5:
                break
            routes[i] = _swap_improve(routes[i], dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 5)
        
        # Phase 2: Inter-route moves
        if time.time() < deadline - 5:
            routes = _inter_route_move(routes, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon, deadline - 5)
    
    # Final verification
    valid_routes = []
    for r in routes:
        if not r:
            continue
        # Re-check feasibility
        feasible, _ = _fast_check_route(r, dist, demands, ready, due, service, n_customers, battery_max, capacity, speed, recharge_rate, horizon)
        if feasible:
            valid_routes.append(r)
    
    all_served = set()
    for r in valid_routes:
        all_served.update(r)
    
    missing = [c for c in range(1, n_customers + 1) if c not in all_served]
    
    for c in missing:
        valid_routes.append([c])
    
    return valid_routes


def plan(instance):
    """Main entry point. Returns routes for the instance."""
    start_time = time.time()
    per_instance_time = 40
    
    routes = _solve_instance(instance, per_instance_time)
    
    return routes
# EVOLVE-BLOCK-END
