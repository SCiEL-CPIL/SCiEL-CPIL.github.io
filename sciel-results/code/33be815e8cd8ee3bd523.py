"""EV routing with time windows - optimized solver."""
import numpy as np
from math import sqrt
import time

def _dist_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2)
            D[i][j] = d
            D[j][i] = d
    return D

def _eval_route(route, D, instance):
    """Evaluate a single route. Returns (distance, feasible, time, battery_ok)."""
    n_customers = instance["n_customers"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    
    # Check capacity
    total_demand = sum(demands[v] for v in route if v <= n_customers)
    if total_demand > capacity:
        return 0, False, 0, False
    
    t = 0.0
    battery = battery_max
    dist = 0.0
    prev = 0  # depot
    
    for v in route:
        d = D[prev][v]
        dist += d
        t += d / speed
        battery -= d
        
        if battery < -1e-9:
            return dist, False, t, False
        
        if v in stations:
            # Recharge: time = (battery - 0) * recharge_rate? No: station adds (battery - energy)*recharge_rate
            # Actually: "a station adds (battery - energy) * recharge_rate time and refills"
            # energy here seems to be 0 (empty), so time = battery * recharge_rate? 
            # Re-reading: "adds (battery - energy) * recharge_rate time and refills"
            # I think 'energy' is the energy consumed to get here, but battery is current level.
            # Let me re-read: "distance d uses d battery and d/speed time; battery never below 0"
            # "a station adds (battery - energy) * recharge_rate time and refills"
            # I think this means: at a station, the time added is (current_battery - 0) * recharge_rate? 
            # No wait - maybe 'energy' refers to something else. Let me interpret: 
            # The station charges from current battery level to full. The time = (battery_max - battery) / recharge_rate?
            # But it says "(battery - energy) * recharge_rate". If energy=0, then time = battery * recharge_rate.
            # Hmm, that's odd. Let me just use: time_to_charge = (battery_max - battery) / recharge_rate
            # Actually re-reading more carefully: "adds (battery - energy) * recharge_rate time"
            # Maybe 'energy' is a typo or means the energy used. Let me try: charge_time = (battery_max - battery) * recharge_rate
            # But recharge_rate is 0.5, and if battery is 100, that's 50 time units to charge 200 units? That seems slow.
            # Alternative: charge_time = (battery_max - battery) / recharge_rate = 200/0.5 = 400? Too slow.
            # Let me try the literal interpretation: time_added = battery * recharge_rate (if energy=0)
            # With battery=100, rate=0.5: time = 50. Then battery becomes full (300).
            # I'll go with: charge_time = (battery_max - battery) * recharge_rate
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = battery_max
        else:
            # Customer
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return dist, False, t, False
            t += service[v]
        
        prev = v
    
    # Return to depot
    d = D[prev][0]
    dist += d
    t += d / speed
    battery -= d
    
    if battery < -1e-9:
        return dist, False, t, False
    if t > horizon + 1e-9:
        return dist, False, t, False
    
    return dist, True, t, True

def _solve_instance(instance, time_limit):
    n_customers = instance["n_customers"]
    coords = instance["coords"]
    D = _dist_matrix(coords)
    capacity = instance["capacity"]
    demands = instance["demands"]
    stations = instance["stations"]
    
    customers = list(range(1, n_customers + 1))
    
    # Determine number of routes based on total demand
    total_demand = sum(demands[c] for c in customers)
    n_routes = max(3, int(np.ceil(total_demand / capacity)) + 1)
    n_routes = min(n_routes, 8)
    
    best_routes = None
    best_dist = float('inf')
    
    deadline = time.time() + time_limit
    
    # Phase 1: K-Means clustering for initial routes
    # Assign customers to n_routes clusters
    np.random.seed(42)
    
    for restart in range(20):
        if time.time() > deadline - 5:
            break
        
        # K-means style clustering
        # Initialize centers randomly from customers
        centers_idx = np.random.choice(customers, n_routes, replace=False)
        centers = [np.array(coords[centers_idx[i]]) for i in range(n_routes)]
        
        clusters = [[] for _ in range(n_routes)]
        for _ in range(30):
            # Assign
            new_clusters = [[] for _ in range(n_routes)]
            for c in customers:
                pc = np.array(coords[c])
                dists = [np.linalg.norm(pc - cc) for cc in centers]
                k = int(np.argmin(dists))
                new_clusters[k].append(c)
            
            # Check convergence
            if all(new_clusters[i] == clusters[i] for i in range(n_routes) if clusters[i]):
                break
            clusters = new_clusters
            
            # Update centers
            for i in range(n_routes):
                if clusters[i]:
                    pts = np.array([coords[c] for c in clusters[i]])
                    centers[i] = pts.mean(axis=0)
        
        # Filter empty clusters
        active_clusters = [c for c in clusters if c]
        if len(active_clusters) < 2:
            continue
        
        # Build routes from clusters
        routes = []
        for cluster in active_clusters:
            # Sort by angle from depot or use nearest neighbor
            route = _build_route(cluster, D, instance)
            if route is not None:
                routes.append(route)
        
        # Verify all customers covered
        all_served = sum(len(r) for r in routes) == n_customers
        if not all_served:
            # Add missing customers
            served = set()
            for r in routes:
                for v in r:
                    if v <= n_customers:
                        served.add(v)
            missing = [c for c in customers if c not in served]
            for m in missing:
                # Insert into best route
                best_idx = 0
                best_cost = float('inf')
                for i, r in enumerate(routes):
                    cost = _insert_cost(m, r, D, instance)
                    if cost < best_cost:
                        best_cost = cost
                        best_idx = i
                routes[best_idx].append(m)
        
        # Evaluate
        total_dist = 0
        feasible = True
        for r in routes:
            d, ok, t, b = _eval_route(r, D, instance)
            total_dist += d
            if not ok:
                feasible = False
                break
        
        if feasible and total_dist < best_dist:
            best_dist = total_dist
            best_routes = [list(r) for r in routes]
        
        # Phase 2: Local search
        if feasible:
            routes = _local_search(routes, D, instance, deadline)
            total_dist = 0
            feasible2 = True
            for r in routes:
                d, ok, t, b = _eval_route(r, D, instance)
                total_dist += d
                if not ok:
                    feasible2 = False
                    break
            if feasible2 and total_dist < best_dist:
                best_dist = total_dist
                best_routes = [list(r) for r in routes]
    
    if best_routes is None:
        # Fallback: one customer per route
        best_routes = [[c] for c in customers]
    
    return best_routes

def _build_route(cluster, D, instance):
    """Build a feasible route from a set of customers using nearest neighbor with time windows."""
    n_customers = instance["n_customers"]
    ready = instance["ready"]
    due = instance["due"]
    
    # Sort by due time for initial ordering
    cluster = sorted(cluster, key=lambda c: due[c])
    
    # Try to insert all customers
    route = []
    remaining = list(cluster)
    
    for _ in range(len(remaining)):
        if not remaining:
            break
        best_c = None
        best_pos = None
        best_cost = float('inf')
        
        for c in remaining:
            for pos in range(len(route) + 1):
                cost, feasible = _insert_feasibility(c, pos, route, D, instance)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_c = c
                    best_pos = pos
        
        if best_c is None:
            # Try to find any feasible insertion
            for c in remaining:
                for pos in range(len(route) + 1):
                    cost, feasible = _insert_feasibility(c, pos, route, D, instance)
                    if feasible:
                        best_c = c
                        best_pos = pos
                        break
                if best_c is not None:
                    break
        
        if best_c is None:
            return None
        
        route.insert(best_pos, best_c)
        remaining.remove(best_c)
    
    # Verify full route
    d, ok, t, b = _eval_route(route, D, instance)
    if not ok:
        return None
    
    return route

def _insert_feasibility(c, pos, route, D, instance):
    """Check if inserting customer c at position pos in route is feasible. Return (cost, feasible)."""
    n_customers = instance["n_customers"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    demands = instance["demands"]
    stations = set(instance["stations"])
    
    # Check capacity
    total_demand = sum(demands[v] for v in route if v <= n_customers) + demands[c]
    if total_demand > instance["capacity"]:
        return float('inf'), False
    
    # Simulate route with insertion
    new_route = route[:pos] + [c] + route[pos:]
    t = 0.0
    battery = battery_max
    prev = 0
    
    for v in new_route:
        d = D[prev][v]
        t += d / speed
        battery -= d
        if battery < -1e-9:
            return float('inf'), False
        
        if v in stations:
            charge_time = (battery_max - battery) * recharge_rate
            t += charge_time
            battery = battery_max
        else:
            if t < ready[v]:
                t = ready[v]
            if t > due[v] + 1e-9:
                return float('inf'), False
            t += service[v]
        prev = v
    
    d = D[prev][0]
    t += d / speed
    battery -= d
    if battery < -1e-9 or t > horizon + 1e-9:
        return float('inf'), False
    
    # Cost: change in distance
    if pos == 0:
        old_cost = D[0][route[0]] if route else 0
        new_cost = D[0][c] + (D[c][route[0]] if route else D[c][0])
    elif pos == len(route):
        old_cost = D[route[-1]][0] if route else D[0][0]
        new_cost = D[route[-1]][c] + D[c][0] if route else D[0][c] + D[c][0]
    else:
        old_cost = D[route[pos-1]][route[pos]]
        new_cost = D[route[pos-1]][c] + D[c][route[pos]]
    
    cost = new_cost - old_cost
    return cost, True

def _insert_cost(c, route, D, instance):
    """Find best insertion cost for customer c into route."""
    best_cost = float('inf')
    for pos in range(len(route) + 1):
        cost, feasible = _insert_feasibility(c, pos, route, D, instance)
        if feasible and cost < best_cost:
            best_cost = cost
    return best_cost

def _local_search(routes, D, instance, deadline):
    """Apply 2-opt, relocate, and swap moves."""
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    customers_in_routes = []
    for r in routes:
        for v in r:
            if v <= n_customers and v not in stations:
                customers_in_routes.append(v)
    
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        
        # 2-opt within each route
        for i, route in enumerate(routes):
            new_route = _two_opt(route, D, instance)
            if new_route is not None:
                d1, ok1, _, _ = _eval_route(route, D, instance)
                d2, ok2, _, _ = _eval_route(new_route, D, instance)
                if ok2 and d2 < d1 - 1e-6:
                    routes[i] = new_route
                    improved = True
        
        # Relocate: move customer from one route to another
        for i in range(len(routes)):
            for j in range(len(routes)):
                if i == j:
                    continue
                for k in range(len(routes[i])):
                    c = routes[i][k]
                    if c in stations or c > n_customers:
                        continue
                    # Try removing from i and inserting into j
                    route_i_without = routes[i][:k] + routes[i][k+1:]
                    if not route_i_without:
                        continue
                    d_i_old, ok_i_old, _, _ = _eval_route(routes[i], D, instance)
                    d_i_new, ok_i_new, _, _ = _eval_route(route_i_without, D, instance)
                    if not ok_i_new:
                        continue
                    
                    # Find best insertion into j
                    best_pos = None
                    best_cost = float('inf')
                    for pos in range(len(routes[j]) + 1):
                        cost, feasible = _insert_feasibility(c, pos, route_i_without if False else routes[j], D, instance)
                        # Actually need to check the combined effect
                        new_j = routes[j][:pos] + [c] + routes[j][pos:]
                        d_j_new, ok_j_new, _, _ = _eval_route(new_j, D, instance)
                        if ok_j_new:
                            d_j_old, _, _, _ = _eval_route(routes[j], D, instance)
                            total_old = d_i_old + d_j_old
                            total_new = d_i_new + d_j_new
                            if total_new < total_old - 1e-6:
                                best_pos = pos
                                best_cost = total_new - total_old
                                break
                    if best_pos is not None:
                        routes[i] = route_i_without
                        routes[j] = routes[j][:best_pos] + [c] + routes[j][best_pos:]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        # Swap: exchange two customers between routes
        for i in range(len(routes)):
            for j in range(i+1, len(routes)):
                for k in range(len(routes[i])):
                    c1 = routes[i][k]
                    if c1 in stations or c1 > n_customers:
                        continue
                    for l in range(len(routes[j])):
                        c2 = routes[j][l]
                        if c2 in stations or c2 > n_customers:
                            continue
                        new_i = list(routes[i])
                        new_j = list(routes[j])
                        new_i[k] = c2
                        new_j[l] = c1
                        d_i_old, ok_i_old, _, _ = _eval_route(routes[i], D, instance)
                        d_j_old, ok_j_old, _, _ = _eval_route(routes[j], D, instance)
                        d_i_new, ok_i_new, _, _ = _eval_route(new_i, D, instance)
                        d_j_new, ok_j_new, _, _ = _eval_route(new_j, D, instance)
                        if ok_i_new and ok_j_new:
                            if d_i_new + d_j_new < d_i_old + d_j_old - 1e-6:
                                routes[i] = new_i
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

def _two_opt(route, D, instance):
    """Apply 2-opt improvement to a single route."""
    if len(route) < 3:
        return None
    
    best_route = list(route)
    best_dist, _, _, _ = _eval_route(best_route, D, instance)
    improved = True
    
    while improved:
        improved = False
        for i in range(len(best_route) - 1):
            for j in range(i + 1, len(best_route)):
                new_route = best_route[:i+1] + best_route[i+1:j+1][::-1] + best_route[j+1:]
                d, ok, t, b = _eval_route(new_route, D, instance)
                if ok and d < best_dist - 1e-6:
                    best_route = new_route
                    best_dist = d
                    improved = True
        if not improved:
            break
    
    return best_route

def plan(instance):
    start_time = time.time()
    time_limit = 135  # Leave 15s safety margin
    
    # Solve single instance
    routes = _solve_instance(instance, time_limit)
    
    return routes
