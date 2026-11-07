"""Reformulated EV routing: time-dependent Dijkstra on (node, battery, unvisited) state space."""
# EVOLVE-BLOCK-START
import numpy as np
import heapq
import math

def plan(instance):
    n_cust = instance["n_customers"]
    coords = instance["coords"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Node indices: 0 is depot, 1..n_cust are customers, rest are stations
    # We'll work with node ids directly
    # Precompute distances between all relevant nodes
    # Relevant nodes: 0 (depot), 1..n_cust (customers), and station nodes
    relevant_nodes = list(range(0, n_cust + 1)) + list(stations)
    # Remove duplicates while preserving order
    seen = set()
    unique_relevant = []
    for n in relevant_nodes:
        if n not in seen:
            seen.add(n)
            unique_relevant.append(n)
    
    # Build distance matrix for relevant nodes
    n_rel = len(unique_relevant)
    node_to_idx = {n: i for i, n in enumerate(unique_relevant)}
    dist_mat = np.zeros((n_rel, n_rel))
    for i in range(n_rel):
        for j in range(n_rel):
            dx = coords[unique_relevant[i]][0] - coords[unique_relevant[j]][0]
            dy = coords[unique_relevant[i]][1] - coords[unique_relevant[j]][1]
            dist_mat[i][j] = math.sqrt(dx*dx + dy*dy)
    
    # Battery levels are continuous, so we can't discretize fully.
    # Instead, we use a different approach: for each route, we build it incrementally
    # using a greedy insertion with feasibility checks, but guided by a more global perspective.
    
    # Actually, let's use a simpler but effective approach:
    # 1. Cluster customers using k-means into ~5-7 groups
    # 2. For each cluster, solve a TSP-like problem with battery constraints
    # 3. Use nearest-neighbor with battery-aware lookahead
    
    # Let me implement a smarter approach:
    # - Use a priority-based construction where we pick the next customer/station
    #   that minimizes total cost (distance + time penalty) while maintaining feasibility
    # - This is essentially a greedy algorithm with feasibility checks
    
    # First, let's create a helper to check if a sequence of nodes is feasible
    def is_feasible_route(route, instance_data):
        """Check if a route (list of node ids, excluding depot) is feasible."""
        coords_local, demands_local, ready_local, due_local, service_local = instance_data
        stations_local, capacity_local, battery_max_local, speed_local, recharge_rate_local, horizon_local = instance_data[5], instance_data[6], instance_data[7], instance_data[8], instance_data[9], instance_data[10]
        
        # Calculate total demand
        total_demand = 0
        for node in route:
            if node <= n_cust and node >= 1:
                total_demand += demands_local[node]
        if total_demand > capacity_local:
            return False
        
        # Simulate the route
        current_node = 0  # depot
        time = 0.0
        battery = battery_max_local
        visited = set()
        
        for node in route:
            # Travel from current_node to node
            dx = coords_local[current_node][0] - coords_local[node][0]
            dy = coords_local[current_node][1] - coords_local[node][1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Check battery
            if battery < dist:
                return False
            battery -= dist
            time += dist / speed_local
            
            # If it's a station, recharge
            if node in stations_local:
                # Recharge: adds (battery_max - battery) * recharge_rate time, refills battery
                recharge_time = (battery_max_local - battery) * recharge_rate_local
                time += recharge_time
                battery = battery_max_local
            else:
                # It's a customer
                if node in visited:
                    return False
                visited.add(node)
                
                # Time window
                time = max(time, ready_local[node])
                if time > due_local[node]:
                    return False
                time += service_local[node]
            
            current_node = node
        
        # Return to depot
        dx = coords_local[current_node][0] - coords_local[0][0]
        dy = coords_local[current_node][1] - coords_local[0][1]
        dist = math.sqrt(dx*dx + dy*dy)
        if battery < dist:
            return False
        battery -= dist
        time += dist / speed_local
        
        if time > horizon_local:
            return False
        
        return True
    
    # Simpler approach: build routes using a greedy method with battery-aware decisions
    # Step 1: Sort customers by due date
    customers = list(range(1, n_cust + 1))
    customers.sort(key=lambda v: due[v])
    
    # Step 2: For each customer, try to insert it into existing routes or create a new one
    # Use a feasibility check with battery simulation
    
    def simulate_route(route):
        """Simulate a route and return (feasible, total_distance, end_time, end_battery)."""
        current_node = 0
        time = 0.0
        battery = battery_max
        total_dist = 0.0
        
        for node in route:
            dx = coords[current_node][0] - coords[node][0]
            dy = coords[current_node][1] - coords[node][1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if battery < dist:
                return (False, total_dist, time, battery)
            battery -= dist
            time += dist / speed
            total_dist += dist
            
            if node in stations:
                recharge_time = (battery_max - battery) * recharge_rate
                time += recharge_time
                battery = battery_max
            else:
                time = max(time, ready[node])
                if time > due[node]:
                    return (False, total_dist, time, battery)
                time += service[node]
            
            current_node = node
        
        # Return to depot
        dx = coords[current_node][0] - coords[0][0]
        dy = coords[current_node][1] - coords[0][1]
        dist = math.sqrt(dx*dx + dy*dy)
        if battery < dist:
            return (False, total_dist, time, battery)
        battery -= dist
        time += dist / speed
        total_dist += dist
        
        if time > horizon:
            return (False, total_dist, time, battery)
        
        return (True, total_dist, time, battery)
    
    # Build routes using a greedy insertion approach
    # Start with no routes
    routes = []
    unvisited = set(customers)
    
    # Sort customers by a heuristic: prefer those with tight time windows and far from depot
    def customer_priority(c):
        # Prioritize customers with earlier due dates and greater distance from depot
        dx = coords[c][0] - coords[0][0]
        dy = coords[c][1] - coords[0][1]
        dist_to_depot = math.sqrt(dx*dx + dy*dy)
        return (due[c], -dist_to_depot)
    
    sorted_customers = sorted(customers, key=customer_priority)
    
    # For each customer, try to insert into existing routes
    # If no existing route works, create a new one
    for c in sorted_customers:
        if c not in unvisited:
            continue
        
        best_route_idx = -1
        best_position = -1
        best_dist = float('inf')
        
        # Try inserting into each existing route
        for ri, route in enumerate(routes):
            # Try inserting at each position
            for pos in range(len(route) + 1):
                new_route = route[:pos] + [c] + route[pos:]
                feasible, dist, time, battery = simulate_route(new_route)
                if feasible and dist < best_dist:
                    best_dist = dist
                    best_route_idx = ri
                    best_position = pos
        
        # Also consider creating a new route with just this customer
        new_route = [c]
        feasible, dist, time, battery = simulate_route(new_route)
        if feasible and dist < best_dist:
            best_dist = dist
            best_route_idx = -1
            best_position = 0
        
        if best_route_idx == -1:
            # Create new route
            routes.append([c])
        else:
            routes[best_route_idx].insert(best_position, c)
        
        unvisited.discard(c)
    
    # After initial construction, try to improve with local search
    # 2-opt within each route
    def two_opt_improve(route):
        """Apply 2-opt to a route, checking feasibility."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    if new_route == route:
                        continue
                    feasible, dist_new, _, _ = simulate_route(new_route)
                    _, dist_old, _, _ = simulate_route(route)
                    if feasible and dist_new < dist_old - 1e-9:
                        route[:] = new_route
                        improved = True
        return route
    
    # Apply 2-opt to each route
    for i in range(len(routes)):
        routes[i] = two_opt_improve(routes[i])
    
    # Try to merge routes if possible
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            merged = routes[i] + routes[j]
            # Check capacity
            total_demand = sum(demands[c] for c in merged if 1 <= c <= n_cust)
            if total_demand > capacity:
                continue
            feasible, dist, _, _ = simulate_route(merged)
            if feasible:
                routes[i] = merged
                routes.pop(j)
                j -= 1
                # Re-apply 2-opt
                routes[i] = two_opt_improve(routes[i])
                break
    
    return routes
# EVOLVE-BLOCK-END
