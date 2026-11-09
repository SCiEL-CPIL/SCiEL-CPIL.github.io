"""
EV Routing Solver: Cluster-Then-Sequence Strategy
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math

def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Parse Input
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = float(instance["capacity"])
    battery = float(instance["battery"])
    speed = float(instance["speed"])
    recharge_rate = float(instance["recharge_rate"])
    horizon = float(instance["horizon"])
    
    n_nodes = len(coords)
    customers = list(range(1, n_customers + 1))
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = float(np.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1]))
            dist[i, j] = d
            dist[j, i] = d

    def route_cost(route):
        """Calculate total distance for a route (depot to first, ... last to depot)."""
        if not route:
            return 0.0
        cost = float(dist[0, route[0]])
        for i in range(len(route) - 1):
            cost += float(dist[route[i], route[i+1]])
        cost += float(dist[route[-1], 0])
        return cost

    def check_feasibility(route):
        """
        Check if a route is feasible regarding:
        - Capacity
        - Time Windows
        - Battery
        - Horizon
        Returns (is_feasible, total_time, total_distance)
        """
        if not route:
            return True, 0.0, 0.0
        
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity + 1e-9:
            return False, 0.0, 0.0
            
        t = 0.0
        bat = battery
        prev = 0
        total_dist = 0.0
        
        for v in route:
            d = float(dist[prev, v])
            bat -= d
            t += d / speed
            total_dist += d
            
            if bat < -1e-9:
                return False, 0.0, 0.0
                
            if v in stations:
                # Recharge
                if bat < 0:
                    bat = 0.0
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = battery
            else:
                # Customer
                if t < ready[v]:
                    t = float(ready[v])
                if t > float(due[v]) + 1e-9:
                    return False, 0.0, 0.0
                t += float(service[v])
                
            prev = v
            
        # Return to depot
        d = float(dist[prev, 0])
        bat -= d
        t += d / speed
        total_dist += d
        
        if bat < -1e-9 or t > horizon + 1e-9:
            return False, 0.0, 0.0
            
        return True, t, total_dist

    def optimize_route_order(route):
        """
        Optimize the order of nodes in a single route using 2-opt.
        Only considers customer nodes. Stations are inserted optimally if needed,
        but for simplicity in this phase, we assume stations are not in the initial
        sequence or we handle them by treating them as 'optional stops' that don't
        have time windows but consume battery. 
        However, the problem states stations are separate nodes. 
        Strategy: 
        1. Separate customers and stations from the route.
        2. Optimize customer order using 2-opt (ignoring stations initially).
        3. Insert stations at positions where battery would drop too low.
        """
        if len(route) <= 1:
            return route
        
        # Separate nodes
        custs = [v for v in route if v not in stations]
        stats = [v for v in route if v in stations]
        
        # Initial order: sorted by due date (earliest first) or nearest neighbor
        # Let's use nearest neighbor from depot as base
        current = [0]
        remaining = custs[:]
        while remaining:
            last = current[-1]
            best_v = None
            best_d = float('inf')
            for v in remaining:
                d = dist[last, v]
                if d < best_d:
                    best_d = d
                    best_v = v
            if best_v is None: break
            current.append(best_v)
            remaining.remove(best_v)
        
        # 2-Opt on customers only
        improved = True
        iterations = 0
        while improved and iterations < 50:
            improved = False
            iterations += 1
            n = len(current)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment i+1 to j
                    old_seg = current[i+1:j+1]
                    new_seg = old_seg[::-1]
                    new_route = current[:i+1] + new_seg + current[j+1:]
                    
                    # Quick distance check
                    # Cost change:
                    # Old: dist[i, i+1] + dist[j, j+1] (if exists)
                    # New: dist[i, j] + dist[i+1, j+1] (if exists)
                    # Note: current[0] is depot.
                    
                    # We need to check feasibility too, which is expensive.
                    # For 2-opt in VRP, we usually accept if distance decreases and then check feasibility.
                    # If infeasible, we might need to repair (insert station).
                    
                    # Let's just compute full cost for simplicity in this small loop
                    # But we need to handle stations.
                    # Since we separated stations, we just optimize customer sequence.
                    # Feasibility of customer sequence (ignoring stations) might be false due to battery.
                    # We will insert stations later.
                    
                    # Calculate distance delta for customer sequence
                    # Nodes: 0, c1, c2, ..., ck
                    # If i=0, prev is 0.
                    
                    # Let's just rebuild and check distance
                    # Actually, 2-opt on a list of nodes
                    d_old = 0.0
                    if i > 0:
                        d_old += dist[current[i-1], current[i]]
                    else:
                        d_old += dist[0, current[i]]
                        
                    if j < n - 1:
                        d_old += dist[current[j], current[j+1]]
                    else:
                        d_old += dist[current[j], 0]
                        
                    d_new = 0.0
                    if i > 0:
                        d_new += dist[current[i-1], new_seg[0]]
                    else:
                        d_new += dist[0, new_seg[0]]
                        
                    if j < n - 1:
                        d_new += dist[new_seg[-1], current[j+1]]
                    else:
                        d_new += dist[new_seg[-1], 0]
                        
                    if d_new < d_old - 1e-9:
                        current = new_route
                        improved = True
                        
        # Now we have an optimized customer sequence.
        # We need to insert stations to ensure battery feasibility.
        # Strategy: Simulate the route. If battery drops below 0 (or margin), insert nearest station.
        
        final_route = []
        bat = battery
        t = 0.0
        prev = 0
        
        # We iterate through the optimized customer list
        # But we might need to insert stations.
        # This is complex. A simpler approach for this step:
        # Just return the customer sequence. The main loop will handle station insertion
        # or we assume the initial greedy construction added stations.
        
        # For this "Cluster-Then-Sequence", we will assume the cluster formation
        # already ensured basic feasibility or we repair in the main loop.
        
        return current

    def insert_stations_if_needed(route):
        """
        Given a route of customers, insert stations where necessary to keep battery > 0.
        """
        if not route:
            return route
            
        new_route = []
        bat = battery
        t = 0.0
        prev = 0
        
        for v in route:
            # Check if we can go from prev to v
            d = float(dist[prev, v])
            bat_after = bat - d
            
            if bat_after < 0:
                # Need to charge
                # Find nearest station to prev (or v?)
                # Ideally, go to a station, charge, then go to v.
                # To minimize distance, pick station S that minimizes dist[prev, S] + dist[S, v]
                # Subject to battery reaching S: bat - dist[prev, S] >= 0
                best_s = None
                best_cost = float('inf')
                for s in stations:
                    d_ps = float(dist[prev, s])
                    if bat - d_ps >= -1e-9:
                        # Can reach station
                        # Cost is dist[prev, s] + dist[s, v]
                        # Plus recharge time impact? We prioritize distance for now.
                        cost = d_ps + float(dist[s, v])
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
                            
                if best_s is None:
                    # Cannot reach any station from prev?
                    # This means the route is infeasible.
                    # Return empty or mark as invalid.
                    return None 
                
                # Go to station
                d_ps = float(dist[prev, best_s])
                bat -= d_ps
                t += d_ps / speed
                
                # Charge
                charge_time = (battery - bat) * recharge_rate
                t += charge_time
                bat = battery
                
                new_route.append(best_s)
                prev = best_s
                
                # Now go to v
                d_sv = float(dist[prev, v])
                bat -= d_sv
                t += d_sv / speed
                
            else:
                # Go directly
                bat = bat_after
                t += d / speed
                
            # Service at v
            if v not in stations:
                if t < ready[v]:
                    t = float(ready[v])
                t += float(service[v])
            
            new_route.append(v)
            prev = v
            
        # Check return to depot
        d_ret = float(dist[prev, 0])
        if bat - d_ret < -1e-9:
            # Need station before returning
            best_s = None
            best_cost = float('inf')
            for s in stations:
                d_ps = float(dist[prev, s])
                if bat - d_ps >= -1e-9:
                    # Can reach station, then depot
                    d_sd = float(dist[s, 0])
                    # Battery after station is full, so d_sd must be <= battery
                    if d_sd <= battery + 1e-9:
                        cost = d_ps + d_sd
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
            if best_s is None:
                return None
            # Insert station
            d_ps = float(dist[prev, best_s])
            bat -= d_ps
            t += d_ps / speed
            charge_time = (battery - bat) * recharge_rate
            t += charge_time
            bat = battery
            new_route.append(best_s)
            prev = best_s
            # Return
            d_sd = float(dist[prev, 0])
            bat -= d_sd
            t += d_sd / speed
            
        else:
            # Return directly
            bat -= d_ret
            t += d_ret / speed
            
        # Final check
        if bat < -1e-9 or t > horizon + 1e-9:
            return None
            
        return new_route

    # --- Main Strategy ---
    
    # 1. Initial Clustering
    # Use k-means like approach.
    # Estimate number of routes.
    # Max demand per route is capacity.
    # Total demand / capacity gives lower bound.
    total_demand = sum(demands[v] for v in customers)
    min_routes = int(math.ceil(total_demand / capacity))
    
    # Also consider battery range.
    # Max distance per route approx battery / 2 (round trip)
    # Let's start with min_routes + 1 to be safe.
    n_routes = min_routes + 1
    
    # Initialize centroids as random customers
    random.seed(42)
    centroids = random.sample(customers, n_routes)
    
    clusters = [[] for _ in range(n_routes)]
    
    # Assign customers to nearest centroid (by distance)
    # Iterate a few times
    for _ in range(10):
        # Clear
        for c in clusters:
            c.clear()
            
        # Assign
        for v in customers:
            best_c = 0
            best_d = float('inf')
            for i, cent in enumerate(centroids):
                d = dist[v, cent]
                if d < best_d:
                    best_d = d
                    best_c = i
            clusters[best_c].append(v)
            
        # Update centroids (mean of coordinates)
        for i, c in enumerate(clusters):
            if not c:
                # If empty, pick a random customer
                centroids[i] = random.choice(customers)
            else:
                xs = [coords[v, 0] for v in c]
                ys = [coords[v, 1] for v in c]
                # Find customer closest to mean
                mean_x = sum(xs) / len(xs)
                mean_y = sum(ys) / len(ys)
                best_v = c[0]
                best_d = float('inf')
                for v in c:
                    d = (coords[v, 0] - mean_x)**2 + (coords[v, 1] - mean_y)**2
                    if d < best_d:
                        best_d = d
                        best_v = v
                centroids[i] = best_v
                
    # 2. Optimize each cluster
    final_routes = []
    for c in clusters:
        if not c:
            continue
        # Optimize order
        ordered_custs = optimize_route_order(c)
        # Insert stations
        route = insert_stations_if_needed(ordered_custs)
        if route is None:
            # Fallback: one customer per route for this cluster
            # This might break "all customers served exactly once" if we just drop it.
            # Instead, we should merge this cluster with another or split it.
            # For now, let's just put them in separate routes (degrade to 1-customer routes)
            for v in c:
                r = insert_stations_if_needed([v])
                if r:
                    final_routes.append(r)
                else:
                    # Even single customer infeasible? Should not happen with battery 300 and coords 0-100.
                    # Max dist is ~141. 141*2 = 282 < 300.
                    final_routes.append([v])
        else:
            final_routes.append(route)
            
    # 3. Local Search: Relocate customers between routes
    # We will try to move a customer from one route to another if it improves distance
    # and maintains feasibility.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_total_dist = sum(route_cost(r) for r in final_routes)
        
        for i in range(len(final_routes)):
            if time.time() > deadline: break
            route_i = final_routes[i]
            for j in range(len(route_i)):
                v = route_i[j]
                if v in stations:
                    continue # Don't move stations in this simple pass
                
                # Try moving v to other routes
                for k in range(len(final_routes)):
                    if k == i:
                        continue
                    route_k = final_routes[k]
                    
                    # Try inserting v at each position in route_k
                    for pos in range(len(route_k) + 1):
                        # Build new routes
                        new_route_i = route_i[:j] + route_i[j+1:]
                        new_route_k = route_k[:pos] + [v] + route_k[pos:]
                        
                        # Check feasibility of new_route_i
                        feas_i, _, _ = check_feasibility(new_route_i)
                        if not feas_i:
                            continue
                            
                        # Check feasibility of new_route_k
                        feas_k, _, _ = check_feasibility(new_route_k)
                        if not feas_k:
                            continue
                        
                        # Calculate new total distance
                        # Delta calculation
                        # Old dist: cost(route_i) + cost(route_k)
                        # New dist: cost(new_route_i) + cost(new_route_k)
                        
                        # Optimization: Calculate delta directly
                        # But for simplicity and safety, compute full cost
                        # Since routes are small (< 10 nodes), this is fast.
                        
                        new_dist_i = route_cost(new_route_i)
                        new_dist_k = route_cost(new_route_k)
                        
                        old_dist_i = route_cost(route_i)
                        old_dist_k = route_cost(route_k)
                        
                        delta = (new_dist_i + new_dist_k) - (old_dist_i + old_dist_k)
                        
                        if delta < -1e-6:
                            final_routes[i] = new_route_i
                            final_routes[k] = new_route_k
                            current_total_dist += delta
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
                
    # 4. Final 2-Opt within routes (including stations this time)
    # We already did 2-opt on customers. Let's do a quick pass on the full route
    # to see if swapping station/customer order helps.
    # Actually, insert_stations_if_needed already placed stations.
    # A full 2-opt on the final route (including stations) might be beneficial.
    
    for i in range(len(final_routes)):
        if time.time() > deadline: break
        route = final_routes[i]
        if len(route) < 2:
            continue
            
        improved_route = True
        while improved_route and time.time() < deadline:
            improved_route = False
            n = len(route)
            for a in range(n):
                for b in range(a + 1, n):
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    feas, _, new_dist = check_feasibility(new_route)
                    if feas:
                        old_dist = route_cost(route)
                        if new_dist < old_dist - 1e-6:
                            route = new_route
                            improved_route = True
            final_routes[i] = route

    return final_routes

# EVOLVE-BLOCK-END
