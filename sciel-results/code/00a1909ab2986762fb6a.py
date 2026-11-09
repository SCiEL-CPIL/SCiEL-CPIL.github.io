import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Extract instance data
    coords = np.array(instance["coords"])
    demands = np.array(instance["demands"])
    ready = np.array(instance["ready"])
    due = np.array(instance["due"])
    service = np.array(instance["service"])
    n_customers = instance["n_customers"]
    stations = instance["stations"]
    capacity = instance["capacity"]
    battery_max = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Node indices: 0 is depot, 1..n_customers are customers, rest are stations
    # Precompute distances
    n_nodes = len(coords)
    dist_matrix = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            d = np.linalg.norm(coords[i] - coords[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    def check_feasibility(route, cluster_customers):
        """
        route: list of node ids (excluding depot at start/end)
        cluster_customers: set of customer ids in this route
        Returns (is_feasible, total_distance, total_time)
        """
        if not route:
            return True, 0.0, 0.0
            
        # Check capacity
        total_demand = sum(demands[c] for c in route if c <= n_customers)
        if total_demand > capacity:
            return False, 0.0, 0.0
            
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        total_dist = 0.0
        
        for node in route:
            d = dist_matrix[curr_node, node]
            curr_time += d / speed
            curr_battery -= d
            total_dist += d
            
            if curr_battery < 0:
                return False, 0.0, 0.0
                
            if node <= n_customers:
                # Customer service
                if curr_time > due[node]:
                    return False, 0.0, 0.0
                curr_time = max(curr_time, ready[node]) + service[node]
            else:
                # Station recharge
                # (battery - energy) * recharge_rate ? 
                # "a station adds (battery - energy) * recharge_rate time and refills"
                # Wait, the prompt says: "a station adds (battery - energy) * recharge_rate time and refills"
                # This phrasing is weird. Usually recharge time = (capacity - current) / rate.
                # Let's re-read carefully: "a station adds (battery - energy) * recharge_rate time and refills"
                # If 'energy' is current battery, then (max - curr) * rate?
                # Or is it (curr - 0) * rate? No, that would mean empty battery takes 0 time.
                # Standard is: time_to_full = (max_battery - current_battery) / recharge_rate.
                # The prompt says: "adds (battery - energy) * recharge_rate time".
                # If battery=300, energy=100. (300-100)*0.5 = 100.
                # If battery=300, energy=0. (300-0)*0.5 = 150.
                # This implies time = (max - curr) * rate.
                # Let's assume this is the formula.
                recharge_time = (battery_max - curr_battery) * recharge_rate
                curr_time += recharge_time
                curr_battery = battery_max
                
        # Return to depot
        d = dist_matrix[curr_node, 0]
        curr_time += d / speed
        curr_battery -= d
        total_dist += d
        
        if curr_battery < 0:
            return False, 0.0, 0.0
        if curr_time > horizon:
            return False, 0.0, 0.0
            
        return True, total_dist, curr_time

    def build_route_greedy(customers_in_cluster):
        """
        Build a route for a set of customers using greedy nearest neighbor with battery check.
        """
        remaining = list(customers_in_cluster)
        route = []
        curr_node = 0
        curr_time = 0.0
        curr_battery = battery_max
        
        while remaining:
            best_node = None
            best_cost = float('inf')
            
            # Check if any customer is directly reachable
            for c in remaining:
                d = dist_matrix[curr_node, c]
                if curr_battery < d:
                    continue
                arr_time = curr_time + d / speed
                if arr_time > due[c]:
                    continue
                # Feasible directly
                cost = d
                if cost < best_cost:
                    best_cost = cost
                    best_node = c
                    best_type = 'customer'
            
            if best_node is not None:
                # Go to customer
                d = dist_matrix[curr_node, best_node]
                curr_time += d / speed
                curr_battery -= d
                route.append(best_node)
                remaining.remove(best_node)
                
                # Service
                curr_time = max(curr_time, ready[best_node]) + service[best_node]
                curr_node = best_node
                continue
            
            # If no customer directly reachable, must go to a station
            # Find best station to visit
            best_station = None
            best_station_cost = float('inf')
            
            for s in stations:
                d1 = dist_matrix[curr_node, s]
                if curr_battery < d1:
                    continue
                # After charging, can we reach any remaining customer?
                # We don't know which one yet, but we want to minimize distance to *some* customer.
                # Heuristic: minimize d1 + min(dist(s, c) for c in remaining if reachable from s)
                # Or just minimize d1?
                # Let's minimize d1 + min distance to any remaining customer from s.
                min_dist_to_customer = float('inf')
                for c in remaining:
                    d2 = dist_matrix[s, c]
                    # Check if reachable from s with full battery
                    if battery_max >= d2:
                        arr_time_from_s = (curr_time + d1/speed + (battery_max - (curr_battery - d1)) * recharge_rate) + d2/speed
                        # Simplified: just check if d2 is small enough
                        if d2 < min_dist_to_customer:
                            min_dist_to_customer = d2
                
                if min_dist_to_customer == float('inf'):
                    continue
                    
                total_cost = d1 + min_dist_to_customer
                if total_cost < best_station_cost:
                    best_station_cost = total_cost
                    best_station = s
            
            if best_station is None:
                # Cannot reach any station or customer. Fail.
                return None
                
            # Go to station
            d1 = dist_matrix[curr_node, best_station]
            curr_time += d1 / speed
            curr_battery -= d1
            route.append(best_station)
            
            # Recharge
            recharge_time = (battery_max - curr_battery) * recharge_rate
            curr_time += recharge_time
            curr_battery = battery_max
            curr_node = best_station
            
        return route

    def two_opt(route):
        """
        Apply 2-Opt local search to a route.
        Returns improved route.
        """
        improved = True
        while improved:
            improved = False
            n = len(route)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Reverse segment route[i..j]
                    new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                    
                    # Check if new route is feasible
                    # We need to check the whole route feasibility
                    feas, dist_new, _ = check_feasibility(new_route, set(new_route))
                    if not feas:
                        continue
                    
                    # Check if it improves distance
                    # We need old distance. Let's compute it once outside or store it.
                    # For simplicity, compute old dist every time? No, too slow.
                    # Let's assume we pass in current best dist.
                    # Actually, 2-Opt on small routes is fast.
                    # Let's just check if new dist < old dist.
                    # We need to compute old dist.
                    feas_old, dist_old, _ = check_feasibility(route, set(route))
                    if dist_new < dist_old - 1e-6:
                        route = new_route
                        improved = True
        return route

    best_solution = None
    best_score = float('inf')
    
    # Try different numbers of clusters
    for K in range(4, 9):
        if time.time() > deadline:
            break
            
        # K-Means Clustering
        # Initialize centers randomly from customers
        customers = list(range(1, n_customers + 1))
        centers = random.sample(customers, K)
        
        for _ in range(20): # K-Means iterations
            # Assign
            clusters = {i: [] for i in range(K)}
            for c in customers:
                dists = [np.linalg.norm(coords[c] - coords[centers[i]]) for i in range(K)]
                min_idx = np.argmin(dists)
                clusters[min_idx].append(c)
            
            # Update
            for i in range(K):
                if clusters[i]:
                    centers[i] = int(np.mean([coords[c] for c in clusters[i]], axis=0).argmin() if False else np.argmin([np.linalg.norm(coords[c] - np.mean([coords[cc] for cc in clusters[i]], axis=0)) for c in customers]))
                    # Simpler: pick customer closest to mean
                    mean_coord = np.mean([coords[c] for c in clusters[i]], axis=0)
                    centers[i] = min(clusters[i], key=lambda c: np.linalg.norm(coords[c] - mean_coord))
            
            # Check convergence
            new_centers = [int(c) for c in centers]
            if new_centers == [int(c) for c in centers]:
                break
            centers = new_centers
        
        # Build routes for each cluster
        routes = []
        feasible = True
        total_dist = 0.0
        
        for i in range(K):
            if not clusters[i]:
                continue
            route = build_route_greedy(clusters[i])
            if route is None:
                feasible = False
                break
            # Apply 2-Opt
            route = two_opt(route)
            routes.append(route)
            
        if not feasible:
            continue
            
        # Check global feasibility and score
        all_customers_served = set()
        for r in routes:
            for node in r:
                if node <= n_customers:
                    if node in all_customers_served:
                        feasible = False
                        break
                    all_customers_served.add(node)
            if not feasible:
                break
                
        if not feasible or len(all_customers_served) != n_customers:
            continue
            
        # Calculate total distance
        for r in routes:
            _, d, _ = check_feasibility(r, set(r))
            total_dist += d
            
        if total_dist < best_score:
            best_score = total_dist
            best_solution = routes

    if best_solution is None:
        # Fallback: one customer per route
        return [[c] for c in range(1, n_customers + 1)]
        
    return best_solution
