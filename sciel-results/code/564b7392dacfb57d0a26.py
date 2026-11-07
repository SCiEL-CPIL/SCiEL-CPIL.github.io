"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.spatial import distance


def plan(instance):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    coords = np.array(instance["coords"], dtype=float)
    n_customers = instance["n_customers"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Precompute distance matrix
    n_nodes = len(coords)
    dist_matrix = distance.cdist(coords, coords, 'euclidean')
    
    # Customer indices
    customers = list(range(1, n_customers + 1))
    
    def simulate_route(route):
        """
        Simulates a route to check feasibility and compute distance.
        Returns (is_feasible, total_distance, total_time).
        Handles charging at stations if needed.
        """
        if not route:
            return True, 0.0, 0.0
            
        # Check capacity
        total_demand = sum(demands[v] for v in route)
        if total_demand > capacity:
            return False, 0.0, 0.0
            
        current_time = 0.0
        current_battery = battery
        current_node = 0
        total_dist = 0.0
        
        # Check if we can even reach the first customer
        # We might need to charge at depot? Depot is not a station usually, but starts full.
        
        for v in route:
            d = dist_matrix[current_node, v]
            
            # Check if we can reach v
            if current_battery < d:
                # Can we charge at current_node?
                if current_node in stations:
                    # Recharge
                    charge_time = (battery - current_battery) * recharge_rate
                    current_time += charge_time
                    current_battery = battery
                    # After recharging, check again
                    if current_battery < d:
                        return False, 0.0, 0.0
                else:
                    # Cannot charge, infeasible
                    return False, 0.0, 0.0
            
            # Travel
            current_time += d / speed
            current_battery -= d
            total_dist += d
            
            # Time window check
            if current_time < ready[v]:
                current_time = ready[v]
            
            if current_time > due[v] + 1e-9:
                return False, 0.0, 0.0
                
            # Service
            current_time += service[v]
            current_node = v
            
        # Return to depot
        d = dist_matrix[current_node, 0]
        
        if current_battery < d:
            if current_node in stations:
                charge_time = (battery - current_battery) * recharge_rate
                current_time += charge_time
                current_battery = battery
                if current_battery < d:
                    return False, 0.0, 0.0
            else:
                return False, 0.0, 0.0
                
        current_time += d / speed
        current_battery -= d
        total_dist += d
        
        if current_time > horizon + 1e-9:
            return False, 0.0, 0.0
            
        return True, total_dist, current_time

    def compute_route_distance(route):
        """Compute total distance of a route including depot legs."""
        if not route:
            return 0.0
        total = dist_matrix[0, route[0]]
        for i in range(len(route) - 1):
            total += dist_matrix[route[i], route[i + 1]]
        total += dist_matrix[route[-1], 0]
        return total

    # --- Construction: Savings Heuristic ---
    
    # Initialize routes: each customer is its own route
    # routes is a list of lists
    # route_map[i] = index of route containing i
    # route_prev[i] = predecessor of i in its route (0 if start)
    # route_next[i] = successor of i in its route (0 if end)
    
    routes = [[i] for i in customers]
    route_index = {i: idx for idx, i in enumerate(customers)}
    prev_node = {i: 0 for i in customers}
    next_node = {i: 0 for i in customers}
    
    # Calculate savings
    # Savings for merging route ending in i and route starting with j
    savings_list = []
    for i in customers:
        for j in customers:
            if i == j: continue
            s = dist_matrix[0, i] + dist_matrix[0, j] - dist_matrix[i, j]
            savings_list.append((s, i, j))
            
    # Sort by savings descending
    savings_list.sort(reverse=True)
    
    def get_route(i):
        """Return the list of nodes in the route containing i."""
        if i == 0: return []
        r = []
        curr = i
        while curr != 0:
            r.append(curr)
            curr = prev_node[curr]
        r.reverse()
        return r

    def try_merge(i, j):
        """
        Try to merge route ending in i and route starting with j.
        i is the last node of route R1.
        j is the first node of route R2.
        """
        if i == 0 or j == 0: return False
        if next_node[i] != 0 or prev_node[j] != 0: return False
        
        # Get the two routes
        R1 = get_route(i)
        R2 = get_route(j)
        
        # New route is R1 + R2
        new_route = R1 + R2
        
        # Check feasibility
        feas, dist, time = simulate_route(new_route)
        if not feas:
            return False
            
        # Also check capacity (simulate_route does this, but double check logic)
        # simulate_route checks capacity.
        
        # Merge
        next_node[i] = j
        prev_node[j] = i
        
        # Update route_index for all nodes in R2 to point to R1's index?
        # Actually, we just need to maintain the linked list.
        # We don't strictly need route_index if we use get_route, but it helps for speed.
        # For N=25, get_route is O(N). Merging is O(N^2) total. Fine.
        
        return True

    # Perform merges
    for s, i, j in savings_list:
        if time.time() > deadline:
            break
            
        # Check if i and j are still at the ends of their respective routes
        if next_node[i] != 0 or prev_node[j] != 0:
            continue
            
        if try_merge(i, j):
            pass # Merged successfully

    # Extract final routes
    final_routes = []
    visited = set()
    for i in customers:
        if i in visited:
            continue
        # Find the start of this route
        curr = i
        while prev_node[curr] != 0:
            curr = prev_node[curr]
        
        # Traverse
        route = []
        curr = curr
        while curr != 0:
            route.append(curr)
            visited.add(curr)
            curr = next_node[curr]
        final_routes.append(route)

    # --- Local Search: 2-opt and Or-opt ---
    
    best_routes = final_routes
    best_dist = sum(compute_route_distance(r) for r in best_routes)
    
    def try_improve(route):
        """Apply 2-opt and Or-opt to a single route."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 2-opt
            for i in range(len(route) - 1):
                if time.time() > deadline:
                    return route
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    feas, _, _ = simulate_route(new_route)
                    if feas:
                        old_d = compute_route_distance(route)
                        new_d = compute_route_distance(new_route)
                        if new_d < old_d - 1e-9:
                            route = new_route
                            improved = True
                            break
                if improved:
                    break
            
            # Or-opt (move segments of size 1, 2, 3)
            if not improved:
                for size in [1, 2, 3]:
                    for i in range(len(route) - size + 1):
                        if time.time() > deadline:
                            return route
                        segment = route[i:i+size]
                        rest = route[:i] + route[i+size:]
                        for j in range(len(rest) + 1):
                            new_route = rest[:j] + segment + rest[j:]
                            feas, _, _ = simulate_route(new_route)
                            if feas:
                                old_d = compute_route_distance(route)
                                new_d = compute_route_distance(new_route)
                                if new_d < old_d - 1e-9:
                                    route = new_route
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                        
        return route

    # Apply local search to each route
    for idx, route in enumerate(best_routes):
        if time.time() > deadline:
            break
        best_routes[idx] = try_improve(route)
        
    # Final distance calculation
    final_dist = sum(compute_route_distance(r) for r in best_routes)
    
    return best_routes
# EVOLVE-BLOCK-END
