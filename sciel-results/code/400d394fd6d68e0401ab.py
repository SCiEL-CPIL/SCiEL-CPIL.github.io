import numpy as np
import time
import random
import math

def plan(instance):
    start_time = time.time()
    deadline = 120.0 # Hard limit is 120s, keep a small margin for safety/overhead
    
    # Extract data
    coords = np.array(instance["coords"])
    n_nodes = len(coords)
    n_customers = instance["n_customers"]
    customers = list(range(1, n_customers + 1))
    stations = instance["stations"]
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    capacity = instance["capacity"]
    max_battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    # Distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    D = np.sqrt((diff ** 2).sum(axis=-1))
    
    # Helper: Simulate a route
    # Returns total_distance if feasible, else float('inf')
    def simulate(route):
        if not route:
            return 0.0
        
        current_node = 0
        current_time = 0.0
        current_battery = max_battery
        current_load = 0
        total_dist = 0.0
        
        for cust in route:
            current_load += demands[cust]
            if current_load > capacity:
                return float('inf')
            
            # Check if direct travel is possible
            dist_direct = D[current_node, cust]
            
            if dist_direct <= current_battery:
                travel_time = dist_direct / speed
                arrival_time = current_time + travel_time
                if arrival_time > due[cust]:
                    return float('inf')
                
                start_service = max(arrival_time, ready[cust])
                end_service = start_service + service[cust]
                
                current_time = end_service
                current_battery -= dist_direct
                total_dist += dist_direct
                current_node = cust
            else:
                # Need to charge
                best_station = -1
                best_total_dist = float('inf')
                best_end_time = float('inf')
                
                for s in stations:
                    d1 = D[current_node, s]
                    if d1 > current_battery:
                        continue
                    d2 = D[s, cust]
                    if d2 > max_battery:
                        continue
                    
                    t1 = current_time + d1 / speed
                    # Recharge to full
                    energy_needed = max_battery - current_battery
                    recharge_time = energy_needed * recharge_rate
                    t2 = t1 + recharge_time
                    t3 = t2 + d2 / speed
                    
                    if t3 > due[cust]:
                        continue
                    
                    total_d = d1 + d2
                    if total_d < best_total_dist:
                        best_total_dist = total_d
                        best_station = s
                        best_end_time = t3
                
                if best_station == -1:
                    return float('inf')
                
                total_dist += best_total_dist
                arrival_at_cust = best_end_time
                start_service = max(arrival_at_cust, ready[cust])
                current_time = start_service + service[cust]
                current_battery = max_battery - D[best_station, cust]
                current_node = cust
        
        # Return to depot
        dist_back = D[current_node, 0]
        if dist_back <= current_battery:
            travel_time = dist_back / speed
            arrival_time = current_time + travel_time
            if arrival_time > horizon:
                return float('inf')
            total_dist += dist_back
        else:
            # Need to charge on the way back
            best_station = -1
            best_total_dist = float('inf')
            
            for s in stations:
                d1 = D[current_node, s]
                if d1 > current_battery:
                    continue
                d2 = D[s, 0]
                if d2 > max_battery:
                    continue
                
                t1 = current_time + d1 / speed
                energy_needed = max_battery - current_battery
                recharge_time = energy_needed * recharge_rate
                t2 = t1 + recharge_time
                t3 = t2 + d2 / speed
                
                if t3 > horizon:
                    continue
                
                total_d = d1 + d2
                if total_d < best_total_dist:
                    best_total_dist = total_d
                    best_station = s
            
            if best_station == -1:
                return float('inf')
            
            total_dist += best_total_dist
            
        return total_dist

    # Initial Solution: Start with one big route, then split
    # This is better than 1-customer-per-route because it captures spatial clustering
    # We use a simple greedy splitting strategy based on capacity and time/battery
    
    # Sort customers by angle or just keep as is? 
    # Let's try a simple nearest neighbor construction for a single route first, 
    # but since we have 25 customers, a single route might be infeasible.
    # Instead, let's use a "Savings" like initialization or just random insertion.
    # Given the small size, a randomized constructive heuristic is good.
    
    def construct_initial():
        # Start with empty routes
        routes = []
        unvisited = customers[:]
        random.shuffle(unvisited)
        
        # Greedy insertion: try to insert each customer into existing routes or start new one
        # We prefer inserting into existing routes if feasible to minimize number of routes
        # but we also want to minimize distance.
        
        # Actually, let's just create one route per customer first, then merge.
        # Or better: Start with one route, add customers one by one. If infeasible, start new route.
        # This tends to create long, thin routes which are bad.
        
        # Let's do: Start with random permutation, split into feasible chunks.
        # This is fast and produces decent spatial grouping if we sort by angle or x-coord.
        
        # Sort by x-coordinate to group spatially
        sorted_custs = sorted(customers, key=lambda x: coords[x, 0])
        
        current_route = []
        current_load = 0
        for c in sorted_custs:
            # Try to add c to current_route
            test_route = current_route + [c]
            # Quick check: capacity
            if current_load + demands[c] > capacity:
                if current_route:
                    routes.append(current_route)
                current_route = [c]
                current_load = demands[c]
            else:
                # Check feasibility of adding c
                # We don't know the exact battery/time without simulating, 
                # but we can simulate the whole route so far + c.
                # Since N is small, simulating is cheap.
                if simulate(test_route) < float('inf'):
                    current_route.append(c)
                    current_load += demands[c]
                else:
                    if current_route:
                        routes.append(current_route)
                    current_route = [c]
                    current_load = demands[c]
        
        if current_route:
            routes.append(current_route)
            
        return routes

    routes = construct_initial()
    
    # Calculate initial cost
    def get_total_cost(routes):
        cost = 0
        for r in routes:
            c = simulate(r)
            if c == float('inf'):
                return float('inf')
            cost += c
        return cost

    current_cost = get_total_cost(routes)
    if current_cost == float('inf'):
        # Fallback: 1 customer per route
        routes = [[c] for c in customers]
        current_cost = get_total_cost(routes)
        
    best_routes = [r[:] for r in routes]
    best_cost = current_cost
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    n_iterations = 0
    
    while T > min_T and time.time() < deadline - 1:
        # Pick a move
        # Relocate or Swap
        move_type = random.choice(['relocate', 'swap'])
        
        new_routes = [r[:] for r in routes]
        valid_move = False
        
        if move_type == 'relocate':
            if len(routes) < 2:
                continue
            src_idx = random.randint(0, len(routes) - 1)
            if len(routes[src_idx]) <= 1:
                continue
            
            cust_pos = random.randint(0, len(routes[src_idx]) - 1)
            cust = routes[src_idx][cust_pos]
            
            dest_idx = random.randint(0, len(routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(routes) - 1)
            
            ins_pos = random.randint(0, len(routes[dest_idx]))
            
            routes[src_idx].pop(cust_pos)
            routes[dest_idx].insert(ins_pos, cust)
            valid_move = True
            
        elif move_type == 'swap':
            if len(routes) < 2:
                continue
            src_idx = random.randint(0, len(routes) - 1)
            dest_idx = random.randint(0, len(routes) - 1)
            while dest_idx == src_idx:
                dest_idx = random.randint(0, len(routes) - 1)
            
            if len(routes[src_idx]) == 0 or len(routes[dest_idx]) == 0:
                continue
                
            c1_pos = random.randint(0, len(routes[src_idx]) - 1)
            c2_pos = random.randint(0, len(routes[dest_idx]) - 1)
            
            c1 = routes[src_idx][c1_pos]
            c2 = routes[dest_idx][c2_pos]
            
            routes[src_idx][c1_pos] = c2
            routes[dest_idx][c2_pos] = c1
            valid_move = True
            
        if not valid_move:
            continue
            
        # Evaluate
        new_cost = get_total_cost(routes)
        
        if new_cost == float('inf'):
            # Reject
            routes = new_routes
            T *= cooling
            continue
            
        delta = new_cost - current_cost
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_routes = [r[:] for r in routes]
        else:
            routes = new_routes
            
        T *= cooling
        n_iterations += 1
        
        # Adaptive cooling: cool down faster if no improvement for a while?
        # Keep it simple for now.
        
    return best_routes
