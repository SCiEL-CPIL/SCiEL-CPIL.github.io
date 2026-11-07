import numpy as np
from math import sqrt
from heapq import heappush, heappop

def plan(instance):
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = 1 + n_customers + len(stations)
    cust_nodes = list(range(1, n_customers + 1))
    station_nodes = set(stations)
    
    # Precompute distance matrix
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            D[i, j] = D[j, i] = sqrt(dx*dx + dy*dy)
    
    def check_route(route):
        """Check feasibility and return total distance."""
        time = 0.0
        battery = float(battery_cap)
        load = 0
        prev = 0
        total_dist = 0.0
        
        for node in route:
            d = D[prev, node]
            if battery - d < -1e-9:
                return False, 0.0
            time += d / speed
            battery -= d
            total_dist += d
            
            if 1 <= node <= n_customers:  # customer
                if time > due[node] + 1e-9:
                    return False, 0.0
                time = max(time, ready[node])
                time += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0
            elif node in station_nodes:  # station
                energy_needed = battery_cap - battery
                time += energy_needed * recharge_rate
                battery = battery_cap
            
            prev = node
        
        d = D[prev, 0]
        if battery - d < -1e-9:
            return False, 0.0
        time += d / speed
        total_dist += d
        
        if time > horizon + 1e-9:
            return False, 0.0
        
        return True, total_dist
    
    def dijkstra_optimal_tour(customers_in_route):
        """
        Find optimal tour for a given set of customers using Dijkstra on state space.
        State: (current_node, time_bucket, battery_level, visited_mask)
        We discretize time and battery to make state space manageable.
        """
        if not customers_in_route:
            return [], 0.0
        
        # Check if all customers fit in capacity
        total_demand = sum(demands[c] for c in customers_in_route)
        if total_demand > capacity:
            return None, float('inf')
        
        # For small sets, use exact Dijkstra with continuous values
        # State: (node, time, battery, visited_mask)
        # Use priority queue: (total_distance, node, time, battery, visited_mask, path)
        
        n_c = len(customers_in_route)
        # Map customer node ids to indices 0..n_c-1 for mask
        cust_to_idx = {c: i for i, c in enumerate(customers_in_route)}
        
        # Start state
        start_state = (0, 0.0, float(battery_cap), 0, [0])
        
        # Priority queue: (distance, node, time, battery, mask, path)
        pq = [(0.0, 0, 0.0, float(battery_cap), 0, [0])]
        
        # Best distance for each state
        # State key: (node, time_rounded, battery_rounded, mask)
        # We'll use a dict to store best distance seen
        best_dist = {}
        
        # Discretization: round time to 0.1, battery to 1.0
        TIME_STEP = 0.1
        BATTERY_STEP = 1.0
        
        def state_key(node, time, battery, mask):
            t_round = round(time / TIME_STEP) * TIME_STEP
            b_round = round(battery / BATTERY_STEP) * BATTERY_STEP
            return (node, t_round, b_round, mask)
        
        found = None
        
        while pq:
            dist_so_far, node, time, battery, mask, path = heappop(pq)
            
            # Check if we've reached depot with all customers visited
            if node == 0 and mask == (1 << n_c) - 1:
                # Check return to depot
                d_back = D[0, 0]  # 0
                # Already at depot
                if time <= horizon + 1e-9:
                    if found is None or dist_so_far < found[1]:
                        found = (path[1:], dist_so_far)
                    break
            
            # Check if we've seen a better state
            key = state_key(node, time, battery, mask)
            if key in best_dist and best_dist[key] <= dist_so_far + 1e-9:
                continue
            best_dist[key] = dist_so_far
            
            # Try moving to each unvisited customer
            for c in customers_in_route:
                c_idx = cust_to_idx[c]
                if mask & (1 << c_idx):
                    continue  # already visited
                
                d = D[node, c]
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                
                new_time = time + d / speed
                if new_time > due[c] + 1e-9:
                    continue
                new_time = max(new_time, ready[c])
                new_time += service[c]
                
                if new_time > horizon + 1e-9:
                    continue
                
                new_mask = mask | (1 << c_idx)
                new_dist = dist_so_far + d
                
                # Check if this state is better
                new_key = state_key(c, new_time, new_battery, new_mask)
                if new_key not in best_dist or best_dist[new_key] > new_dist + 1e-9:
                    heappush(pq, (new_dist, c, new_time, new_battery, new_mask, path + [c]))
            
            # Try moving to stations
            for s in station_nodes:
                d = D[node, s]
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                
                new_time = time + d / speed
                energy_needed = battery_cap - new_battery
                new_time += energy_needed * recharge_rate
                new_battery = float(battery_cap)
                
                if new_time > horizon + 1e-9:
                    continue
                
                new_dist = dist_so_far + d
                new_key = state_key(s, new_time, new_battery, mask)
                if new_key not in best_dist or best_dist[new_key] > new_dist + 1e-9:
                    heappush(pq, (new_dist, s, new_time, new_battery, mask, path + [s]))
            
            # Try returning to depot
            if node != 0:
                d = D[node, 0]
                new_battery = battery - d
                if new_battery < -1e-9:
                    continue
                
                new_time = time + d / speed
                if new_time > horizon + 1e-9:
                    continue
                
                new_dist = dist_so_far + d
                new_key = state_key(0, new_time, new_battery, mask)
                if new_key not in best_dist or best_dist[new_key] > new_dist + 1e-9:
                    heappush(pq, (new_dist, 0, new_time, new_battery, mask, path + [0]))
        
        if found is None:
            return None, float('inf')
        return found[0], found[1]
    
    # Step 1: Partition customers into groups using greedy insertion
    def insertion_partition():
        customers_sorted = sorted(cust_nodes, key=lambda v: due[v])
        routes = []
        
        for cust in customers_sorted:
            best_route_idx = -1
            best_pos = -1
            best_cost = float('inf')
            
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [cust] + route[pos:]
                    feasible, cost = check_route(new_route)
                    if feasible and cost < best_cost:
                        best_cost = cost
                        best_route_idx = r_idx
                        best_pos = pos
            
            if best_route_idx >= 0:
                routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
            else:
                routes.append([cust])
        
        return routes
    
    # Step 2: Optimize each route using Dijkstra
    def optimize_routes(routes):
        optimized = []
        for route in routes:
            # Extract customers from route
            customers_in_route = [n for n in route if 1 <= n <= n_customers]
            if not customers_in_route:
                continue
            
            optimal_tour, cost = dijkstra_optimal_tour(customers_in_route)
            if optimal_tour is not None:
                optimized.append(optimal_tour)
            else:
                # Fallback to original route if Dijkstra fails
                optimized.append(route)
        return optimized
    
    # Step 3: Local search - move customers between routes
    def local_search(routes):
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for i in range(len(routes)):
                for j in range(len(routes[i])):
                    cust = routes[i][j]
                    # Try moving to other routes
                    for k in range(len(routes)):
                        if k == i:
                            continue
                        for pos in range(len(routes[k]) + 1):
                            # Remove from i, insert into k
                            new_routes_i = routes[i][:j] + routes[i][j+1:]
                            new_routes_k = routes[k][:pos] + [cust] + routes[k][pos:]
                            
                            # Check feasibility
                            feas_i, cost_i = check_route(new_routes_i) if new_routes_i else (True, 0.0)
                            feas_k, cost_k = check_route(new_routes_k)
                            
                            if feas_i and feas_k:
                                # Compute total cost
                                old_total = 0
                                for r in routes:
                                    _, c = check_route(r) if r else (True, 0.0)
                                    old_total += c
                                
                                new_total = cost_i + cost_k
                                for r_idx, r in enumerate(routes):
                                    if r_idx != i and r_idx != k:
                                        _, c = check_route(r) if r else (True, 0.0)
                                        new_total += c
                                
                                if new_total < old_total - 1e-9:
                                    routes[i] = new_routes_i
                                    routes[k] = new_routes_k
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
    
    # Run the algorithm
    routes = insertion_partition()
    routes = optimize_routes(routes)
    routes = local_search(routes)
    
    # Ensure all customers are served
    served = set()
    for route in routes:
        for node in route:
            if 1 <= node <= n_customers:
                served.add(node)
    
    missing = set(cust_nodes) - served
    for cust in missing:
        routes.append([cust])
    
    return routes
