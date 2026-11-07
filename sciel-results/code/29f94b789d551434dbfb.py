# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt

def plan(instance):
    start_time = time.time()
    DEADLINE = 110.0
    
    coords = np.array(instance["coords"], dtype=np.float64)
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
            
    customers = list(range(1, n_customers + 1))
    # Sort by due date for insertion
    customers.sort(key=lambda v: due[v])
    
    routes = []
    
    def is_feasible(route):
        """Check if a route is feasible. Returns (valid, total_distance)."""
        if not route:
            return True, 0.0
            
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        total_demand = 0.0
        
        for node in route:
            d = dist[curr_node][node]
            total_dist += d
            
            # Check battery before travel
            if curr_battery < d:
                return False, 0.0
            
            curr_battery -= d
            curr_time += d / speed
            
            if node in stations:
                # Charge
                charge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += charge_time
                curr_battery = battery_max
            else:
                # Customer
                total_demand += demands[node]
                if total_demand > capacity:
                    return False, 0.0
                
                arrival_time = curr_time
                service_start = max(arrival_time, ready[node])
                if service_start > due[node]:
                    return False, 0.0
                
                curr_time = service_start + service[node]
                
            curr_node = node
            
        # Return to depot
        d = dist[curr_node][0]
        total_dist += d
        
        if curr_battery < d:
            return False, 0.0
        
        curr_battery -= d
        curr_time += d / speed
        
        if curr_time > horizon:
            return False, 0.0
            
        return True, total_dist

    def route_distance(route):
        if not route:
            return 0.0
        curr = 0
        total = 0.0
        for node in route:
            total += dist[curr][node]
            curr = node
        total += dist[curr][0]
        return total

    def two_opt(route):
        """Improve route using 2-opt. Returns improved route."""
        if len(route) < 3:
            return route
            
        best_route = route[:]
        best_dist, _ = is_feasible(best_route)
        if not best_dist: # is_feasible returns (valid, dist), so check valid
             # Wait, is_feasible returns (bool, float). 
             # If invalid, dist is 0.0. 
             # If valid, dist > 0.
             pass
             
        improved = True
        while improved:
            improved = False
            n = len(best_route)
            for i in range(n - 1):
                if time.time() - start_time > DEADLINE:
                    return best_route
                for j in range(i + 1, n):
                    # Reverse segment i..j
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    valid, new_dist = is_feasible(new_route)
                    if valid and new_dist < best_dist:
                        best_route = new_route
                        best_dist = new_dist
                        improved = True
                        break
                if improved:
                    break
        return best_route

    # Greedy Insertion
    for c in customers:
        if time.time() - start_time > DEADLINE:
            break
            
        best_route_idx = -1
        best_pos = -1
        best_cost = float('inf')
        
        # Try inserting into existing routes
        for r_idx, route in enumerate(routes):
            # Check if adding this customer exceeds capacity
            current_demand = sum(demands[node] for node in route)
            if current_demand + demands[c] > capacity:
                continue
                
            # Try all positions
            for p in range(len(route) + 1):
                # Construct new route
                new_route = route[:p] + [c] + route[p:]
                
                # Quick distance check
                prev_node = route[p-1] if p > 0 else 0
                next_node = route[p] if p < len(route) else 0
                
                # Calculate incremental distance
                old_seg_dist = dist[prev_node][next_node]
                new_seg_dist = dist[prev_node][c] + dist[c][next_node]
                
                # Estimate total distance
                current_dist = route_distance(route)
                est_dist = current_dist - old_seg_dist + new_seg_dist
                
                # If estimated distance is already worse, skip feasibility check (optional optimization)
                # But we must check feasibility.
                valid, actual_dist = is_feasible(new_route)
                
                if valid and actual_dist < best_cost:
                    best_cost = actual_dist
                    best_route_idx = r_idx
                    best_pos = p
                    
        if best_route_idx != -1:
            routes[best_route_idx].insert(best_pos, c)
        else:
            # Start new route
            routes.append([c])
            
    # Local Search
    for r_idx in range(len(routes)):
        if time.time() - start_time > DEADLINE:
            break
        routes[r_idx] = two_opt(routes[r_idx])
        
    return routes
# EVOLVE-BLOCK-END
