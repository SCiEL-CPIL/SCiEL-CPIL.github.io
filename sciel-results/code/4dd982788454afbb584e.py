"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import math

def plan(instance):
    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distances
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = np.sqrt((coords[i,0]-coords[j,0])**2 + (coords[i,1]-coords[j,1])**2)
            dist[i,j] = d
            dist[j,i] = d
    
    customers = list(range(1, n_customers + 1))
    
    # Sort customers by due time for initial ordering
    customers_sorted = sorted(customers, key=lambda v: due[v])
    
    # Greedy construction: build routes by inserting customers
    # Start with empty routes, assign customers one by one
    
    # First, let's try to build a few routes using a nearest-neighbor-like approach
    # with time window and battery constraints
    
    def can_insert(route, cust, pos, time_arr, battery_arr):
        """Check if customer cust can be inserted at position pos in route."""
        # route is a list of node ids (customers only, depot implicit)
        # We need to check the full sequence: depot -> ... -> cust -> ... -> depot
        
        # Build the full sequence with cust inserted at pos
        seq = [0] + route[:pos] + [cust] + route[pos:] + [0]
        
        # Check time windows and battery
        t = 0.0
        bat = battery
        for i in range(len(seq)):
            node = seq[i]
            if i > 0:
                d = dist[seq[i-1], node]
                bat -= d
                if bat < -1e-9:
                    return False
                t += d / speed
            
            if node in stations:
                # Recharge
                energy = bat
                bat = battery
                t += energy * recharge_rate
            elif node > 0 and node <= n_customers:
                # Customer
                t = max(t, ready[node])
                if t > due[node] + 1e-9:
                    return False
                t += service[node]
            # node 0 is depot, no service
            
        # Final check: back at depot by horizon
        if t > horizon + 1e-9:
            return False
        
        return True
    
    def route_distance(route):
        """Calculate total distance of a route."""
        if not route:
            return 0.0
        seq = [0] + route + [0]
        total = 0.0
        for i in range(len(seq)-1):
            total += dist[seq[i], seq[i+1]]
        return total
    
    def is_feasible(route):
        """Check if a route is feasible (time windows, battery, horizon)."""
        seq = [0] + route + [0]
        t = 0.0
        bat = battery
        for i in range(len(seq)):
            node = seq[i]
            if i > 0:
                d = dist[seq[i-1], node]
                bat -= d
                if bat < -1e-9:
                    return False
                t += d / speed
            
            if node in stations:
                energy = bat
                bat = battery
                t += energy * recharge_rate
            elif node > 0 and node <= n_customers:
                t = max(t, ready[node])
                if t > due[node] + 1e-9:
                    return False
                t += service[node]
        
        if t > horizon + 1e-9:
            return False
        
        return True
    
    # Greedy: assign each customer to the best existing route or start a new one
    routes = []
    unassigned = set(customers)
    
    # Start with a few seed routes
    # Try to create initial routes using nearest neighbor
    for _ in range(4):  # 4 initial routes
        route = []
        remaining = list(unassigned)
        if not remaining:
            break
        
        # Start from depot, pick nearest feasible customer
        current = 0
        t = 0.0
        bat = battery
        
        while remaining:
            best_cust = None
            best_dist = float('inf')
            
            for cust in remaining:
                d = dist[current, cust]
                if d >= best_dist:
                    continue
                
                # Check if we can go to cust
                new_bat = bat - d
                if new_bat < -1e-9:
                    continue
                new_t = t + d / speed
                new_t = max(new_t, ready[cust])
                if new_t > due[cust] + 1e-9:
                    continue
                
                # Check if we can return to depot after
                d_back = dist[cust, 0]
                final_bat = new_bat - d_back
                if final_bat < -1e-9:
                    continue
                final_t = new_t + service[cust] + d_back / speed
                if final_t > horizon + 1e-9:
                    continue
                
                best_cust = cust
                best_dist = d
            
            if best_cust is None:
                break
            
            # Move to best_cust
            d = dist[current, best_cust]
            bat -= d
            t += d / speed
            t = max(t, ready[best_cust])
            t += service[best_cust]
            route.append(best_cust)
            remaining.remove(best_cust)
            unassigned.discard(best_cust)
            current = best_cust
        
        if route:
            routes.append(route)
    
    # Assign remaining customers
    for cust in list(unassigned):
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                if can_insert(route, cust, pos, None, None):
                    new_route = route[:pos] + [cust] + route[pos:]
                    cost = route_distance(new_route)
                    if cost < best_cost:
                        best_cost = cost
                        best_route_idx = ri
                        best_pos = pos
        
        if best_route_idx >= 0:
            routes[best_route_idx] = routes[best_route_idx][:best_pos] + [cust] + routes[best_route_idx][best_pos:]
            unassigned.discard(cust)
        else:
            # Start a new route
            routes.append([cust])
            unassigned.discard(cust)
    
    # Local search: 2-opt and relocate within routes
    def two_opt(route):
        """Apply 2-opt improvement to a route."""
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(n-1):
                for j in range(i+2, n):
                    if j - i == 1:
                        continue
                    # Reverse segment i..j
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    if is_feasible(new_route) and route_distance(new_route) < route_distance(route) - 1e-9:
                        route = new_route
                        improved = True
        return route
    
    def relocate(route, cust, new_pos):
        """Relocate customer cust to new position in route."""
        if cust not in route:
            return route
        route.remove(cust)
        route.insert(new_pos, cust)
        return route
    
    # Apply 2-opt to each route
    for i in range(len(routes)):
        routes[i] = two_opt(routes[i])
    
    # Relocate customers between routes
    improved = True
    while improved:
        improved = False
        for ri in range(len(routes)):
            for cust in list(routes[ri]):
                best_dest = -1
                best_pos = -1
                best_cost = float('inf')
                
                current_total = sum(route_distance(r) for r in routes)
                
                for rj in range(len(routes)):
                    if rj == ri:
                        continue
                    for pos in range(len(routes[rj]) + 1):
                        # Check if cust can be inserted into routes[rj] at pos
                        if can_insert(routes[rj], cust, pos, None, None):
                            # Check if routes[ri] without cust is still feasible
                            temp_route = [c for c in routes[ri] if c != cust]
                            if is_feasible(temp_route):
                                new_cost = route_distance(temp_route) + route_distance(routes[rj][:pos] + [cust] + routes[rj][pos:])
                                if new_cost < best_cost:
                                    best_cost = new_cost
                                    best_dest = rj
                                    best_pos = pos
                
                if best_dest >= 0 and best_cost < current_total - 1e-9:
                    routes[ri] = [c for c in routes[ri] if c != cust]
                    routes[best_dest] = routes[best_dest][:best_pos] + [cust] + routes[best_dest][best_pos:]
                    improved = True
                    break
            if improved:
                break
    
    # Final 2-opt pass
    for i in range(len(routes)):
        routes[i] = two_opt(routes[i])
    
    # Verify all customers are served exactly once
    all_customers = []
    for route in routes:
        all_customers.extend(route)
    
    if set(all_customers) != set(customers) or len(all_customers) != len(customers):
        # Fallback: one customer per route
        return [[v] for v in customers_sorted]
    
    return routes

# EVOLVE-BLOCK-END
