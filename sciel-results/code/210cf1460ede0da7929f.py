"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from heapq import heappush, heappop
from itertools import combinations


def _euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def _build_dist_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _euclidean(coords[i], coords[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist


def _simulate_route(route, instance, dist):
    """
    Simulate a route and return (total_distance, is_valid).
    Route is a list of customer/station node ids (excluding depot).
    """
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    total_dist = 0.0
    total_demand = 0
    current_battery = battery_cap
    current_time = 0.0
    current_node = 0  # depot
    
    for node in route:
        d = dist[current_node][node]
        current_time += d / speed
        current_battery -= d
        total_dist += d
        
        if current_battery < -1e-9:
            return total_dist, False
        
        if node in stations:
            # Charging: add (battery - energy) * recharge_rate time? 
            # Wait, task says: "a station adds (battery - energy) * recharge_rate time and refills"
            # This phrasing is slightly ambiguous. Usually charging time is (capacity - current) / rate.
            # Let's re-read carefully: "adds (battery - energy) * recharge_rate time".
            # If battery is current charge and energy is capacity?
            # Or is it (energy_needed) * rate?
            # "refills" implies battery becomes full.
            # Let's assume standard: time += (battery_cap - current_battery) * recharge_rate.
            # The text says: (battery - energy) * recharge_rate.
            # If "battery" means capacity and "energy" means current level?
            # Or "battery" is current level and "energy" is 0?
            # Let's look at the seed code. It doesn't use stations.
            # Let's assume the standard interpretation: Time to charge from current to full.
            # charge_time = (battery_cap - current_battery) * recharge_rate
            # The prompt says: "adds (battery - energy) * recharge_rate time".
            # If battery=capacity, energy=current, then (cap - curr) * rate. This makes sense.
            charge_time = (battery_cap - current_battery) * recharge_rate
            current_time += charge_time
            current_battery = battery_cap
        else:
            # Customer
            total_demand += instance["demands"][node]
            if total_demand > capacity:
                return total_dist, False
            
            ready = instance["ready"][node]
            due = instance["due"][node]
            service = instance["service"][node]
            
            if current_time < ready:
                current_time = ready
            
            if current_time > due + 1e-9:
                return total_dist, False
            
            current_time += service
        
        current_node = node
    
    # Return to depot
    d = dist[current_node][0]
    current_time += d / speed
    current_battery -= d
    total_dist += d
    
    if current_battery < -1e-9:
        return total_dist, False
    
    if current_time > horizon + 1e-9:
        return total_dist, False
    
    return total_dist, True


def _nearest_neighbor_route(customers, instance, dist, start_node=0):
    """
    Build a route using Nearest Neighbor heuristic, respecting constraints.
    Returns a list of node ids.
    """
    remaining = set(customers)
    route = []
    current_node = start_node
    current_time = 0.0
    current_battery = instance["battery"]
    total_demand = 0
    
    while remaining:
        best_node = None
        best_dist = float('inf')
        
        for node in remaining:
            d = dist[current_node][node]
            if d >= best_dist:
                continue
            
            # Check if we can reach this node
            new_battery = current_battery - d
            if new_battery < -1e-9:
                continue
            
            new_time = current_time + d / instance["speed"]
            
            # Check if we need to charge before going to this node?
            # We can only charge at stations. We don't insert stations here directly in NN,
            # but we need to ensure feasibility.
            # For simplicity in NN, we assume we can always reach it if battery allows.
            # If battery is low, we might need a station. 
            # A simple NN might fail. 
            # Let's just pick the nearest feasible one.
            
            if node not in instance["stations"]:
                total_demand_check = total_demand + instance["demands"][node]
                if total_demand_check > instance["capacity"]:
                    continue
                
                ready = instance["ready"][node]
                due = instance["due"][node]
                
                arrival_time = new_time
                if arrival_time < ready:
                    arrival_time = ready
                
                if arrival_time > due + 1e-9:
                    continue
            
            best_node = node
            best_dist = d
        
        if best_node is None:
            # If no node is directly reachable/feasible, we might be stuck.
            # For now, break and let the repair handle it or fail.
            break
        
        # Move to best_node
        d = dist[current_node][best_node]
        current_time += d / instance["speed"]
        current_battery -= d
        total_dist = 0 # Not tracking dist here, just building route
        
        if best_node in instance["stations"]:
            charge_time = (instance["battery"] - current_battery) * instance["recharge_rate"]
            current_time += charge_time
            current_battery = instance["battery"]
        else:
            total_demand += instance["demands"][best_node]
            ready = instance["ready"][best_node]
            if current_time < ready:
                current_time = ready
            current_time += instance["service"][best_node]
        
        route.append(best_node)
        remaining.remove(best_node)
        current_node = best_node
    
    return route


def _insert_customer(route, customer, instance, dist):
    """
    Try to insert a customer into an existing route at the best position.
    Returns (new_route, is_valid, extra_distance) or (None, False, inf)
    """
    n = len(route)
    best_pos = -1
    best_extra_dist = float('inf')
    best_new_route = None
    
    for i in range(n + 1):
        # Insert at position i
        new_route = route[:i] + [customer] + route[i:]
        
        # Check validity
        total_dist, is_valid = _simulate_route(new_route, instance, dist)
        if not is_valid:
            continue
        
        # Calculate extra distance
        # We need to compare with original distance
        # For simplicity, we can just use the total distance difference
        # But we don't have original distance. 
        # Let's compute original distance once outside or pass it in.
        # For now, let's just pick the first valid one or the one with min total dist?
        # Min total dist is better.
        
        if total_dist < best_extra_dist:
            best_extra_dist = total_dist
            best_pos = i
            best_new_route = new_route
    
    if best_pos == -1:
        return None, False, float('inf')
    
    return best_new_route, True, best_extra_dist


def _repair_route(route, instance, dist):
    """
    Try to repair an invalid route by inserting stations or reordering.
    This is a simple heuristic: if battery fails, insert the nearest station before the failing node.
    """
    # This is complex to implement robustly. 
    # For now, we will rely on the construction heuristic to be feasible.
    # If a route is invalid, we might need to split it.
    pass


def plan(instance):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    dist = _build_dist_matrix(coords.tolist())
    
    customers = list(range(1, instance["n_customers"] + 1))
    stations = instance["stations"]
    
    # 1. Initial Solution: Nearest Neighbor
    # Sort customers by due time to prioritize tight windows
    customers_sorted = sorted(customers, key=lambda c: instance["due"][c])
    
    initial_routes = []
    remaining_customers = list(customers_sorted)
    
    while remaining_customers:
        # Start a new route from depot
        route = _nearest_neighbor_route(remaining_customers, instance, dist)
        
        if not route:
            # If NN failed, just take the first customer to ensure progress
            route = [remaining_customers[0]]
        
        # Remove served customers from remaining
        served = set(route)
        remaining_customers = [c for c in remaining_customers if c not in served]
        
        initial_routes.append(route)
    
    # 2. Local Search: 2-Opt and Or-opt within routes, and between routes
    # We will use a simple iterative improvement.
    
    best_routes = [list(r) for r in initial_routes]
    
    # Calculate initial cost
    def total_distance(routes):
        total = 0.0
        for route in routes:
            d, valid = _simulate_route(route, instance, dist)
            if not valid:
                return float('inf')
            total += d
        return total
    
    current_cost = total_distance(best_routes)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try 2-Opt within each route
        for i, route in enumerate(best_routes):
            n = len(route)
            for a in range(n):
                for b in range(a + 1, n):
                    if time.time() > deadline:
                        break
                    
                    # Reverse segment a..b
                    new_route = route[:a] + route[a:b+1][::-1] + route[b+1:]
                    new_cost, valid = _simulate_route(new_route, instance, dist)
                    
                    if valid:
                        # Compare with current cost of this route
                        old_cost, _ = _simulate_route(route, instance, dist)
                        if new_cost < old_cost - 1e-9:
                            best_routes[i] = new_route
                            improved = True
                            current_cost += (new_cost - old_cost)
        
        if time.time() > deadline:
            break
        
        # Try Or-opt: move 1, 2, or 3 nodes from one route to another
        # This is expensive. We'll do a simplified version.
        for i in range(len(best_routes)):
            for j in range(len(best_routes[i])):
                if time.time() > deadline:
                    break
                
                node = best_routes[i][j]
                
                # Try removing node from route i
                new_route_i = best_routes[i][:j] + best_routes[i][j+1:]
                
                # Try inserting into other routes
                for k in range(len(best_routes)):
                    if k == i:
                        continue
                    
                    # Try all positions in route k
                    route_k = best_routes[k]
                    for pos in range(len(route_k) + 1):
                        new_route_k = route_k[:pos] + [node] + route_k[pos:]
                        
                        # Check validity of both new routes
                        valid_i, dist_i = _simulate_route(new_route_i, instance, dist)
                        valid_k, dist_k = _simulate_route(new_route_k, instance, dist)
                        
                        if valid_i and valid_k:
                            # Calculate change in total cost
                            old_dist_i, _ = _simulate_route(best_routes[i], instance, dist)
                            old_dist_k, _ = _simulate_route(best_routes[k], instance, dist)
                            
                            new_total = dist_i + dist_k
                            old_total = old_dist_i + old_dist_k
                            
                            if new_total < old_total - 1e-9:
                                best_routes[i] = new_route_i
                                best_routes[k] = new_route_k
                                improved = True
                                current_cost += (new_total - old_total)
                                break # Move to next node in route i
        
        if time.time() > deadline:
            break
    
    # Final check for validity
    for route in best_routes:
        d, valid = _simulate_route(route, instance, dist)
        if not valid:
            # Fallback to seed solution if local search broke something
            customers = list(range(1, instance["n_customers"] + 1))
            customers.sort(key=lambda v: instance["due"][v])
            return [[v] for v in customers]
    
    return best_routes
# EVOLVE-BLOCK-END
