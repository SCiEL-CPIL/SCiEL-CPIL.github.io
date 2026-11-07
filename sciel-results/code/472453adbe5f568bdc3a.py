"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    coords = np.array(instance["coords"], dtype=float)
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

    n_nodes = len(coords)
    
    # Precompute distance matrix
    dist = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
            dist[i, j] = d
            dist[j, i] = d

    # Map customer IDs to indices 0..n_customers-1 for GA permutation
    # Customer IDs are 1..n_customers
    # We will represent a solution as a permutation of [0, 1, ..., n_customers-1]
    # Splitting this permutation into routes is part of the decoding.
    
    # To make GA effective, we need a good decoder.
    # Strategy: The permutation defines the *global* order of customers.
    # We split this order into routes greedily: start a new route when the current one
    # becomes infeasible (capacity, battery, or time window violation that cannot be fixed by reordering within the route? No, reordering is hard in decoder).
    # Better Strategy for GA: 
    # 1. Permutation of customers.
    # 2. Split points (binary mask or list of lengths).
    # 3. Decoder: For each segment, try to construct a valid route. 
    #    If a segment is invalid, apply local repairs (insert stations, swap neighbors) or drop customers to new routes?
    #    Dropping complicates "visit exactly once".
    
    # Alternative GA Representation:
    # Chromosome: List of routes (variable length).
    # This is messy.
    
    # Let's stick to a standard approach for VRPTW with batteries:
    # 1. Construct initial population using a heuristic (e.g., nearest neighbor with time/battery checks).
    # 2. Evolutionary operators:
    #    - Crossover: OX (Order Crossover) on the sequence of customers, then re-optimize route splits.
    #    - Mutation: Swap, Insert, 2-opt within a route.
    #    - Selection: Tournament.
    
    # However, implementing a robust GA that handles the complex "battery never below 0" and "charging station insertion" inside the fitness function is complex.
    # The current program uses a "check_route" function that returns False if invalid.
    # In a GA, we need a fitness function that penalizes invalidity or repairs it.
    
    # Let's try a different structural approach: **Large Neighborhood Search (LNS)** with a specific focus on **Route Re-optimization**.
    # Actually, the current program already does local search. The stagnation suggests the local search is getting stuck.
    # A GA might escape local minima.
    
    # Let's implement a simple GA.
    # Representation: A single permutation of all customers.
    # Decoding: Split the permutation into routes.
    # How to split? We can use a "split" heuristic:
    #   Start at depot. Add customers from the permutation one by one.
    #   If adding the next customer makes the route infeasible (capacity, battery, time), start a new route.
    #   This is a "sequential" decoding.
    #   BUT: The order in the permutation matters for the intra-route sequence.
    #   So, if we have permutation P, and we split it into segments S1, S2, ..., Sk.
    #   Each segment Si is a list of customers. We then need to optimize the order *within* Si?
    #   If the permutation P already defines the order, then Si is the order.
    #   So the GA optimizes both the global order and the implicit splits.
    
    # Problem: If the greedy split creates an infeasible route (e.g., battery runs out before reaching the customer), the decoder must handle it.
    # We can modify the decoder:
    #   For a segment, simulate the route. If battery < 0 or time > due, try to insert a charging station or swap with previous/next.
    #   If still infeasible, the fitness is penalized heavily.
    
    # Let's implement this.
    
    customers = list(range(1, n_customers + 1))
    
    # Helper: Check if a list of customers (in given order) forms a valid route, 
    # and if not, try to repair by inserting stations or reordering?
    # Reordering in decoder is expensive. Let's assume the permutation provides a good order, 
    # and the decoder just checks feasibility. If infeasible, return high penalty.
    # But we need *valid* solutions to return.
    
    # Better Decoder:
    # 1. Take a permutation.
    # 2. Split into routes based on capacity and simple distance heuristics.
    # 3. For each route, run a local search (2-opt + station insertion) to make it feasible and optimal.
    # 4. If a route cannot be made feasible, merge it with neighbors or split differently?
    
    # This is getting complicated. Let's look at the "Inspiration 1" which scored 1.0.
    # The current program scores 2.19.
    # The task asks for a *different* strategy.
    
    # Let's try **Simulated Annealing** on a permutation representation with a robust decoder.
    # Decoder:
    # 1. Split permutation into segments such that each segment's total demand <= capacity.
    # 2. For each segment, treat it as a route.
    # 3. Optimize the route: 
    #    a. Check time windows and battery.
    #    b. If infeasible, try to insert charging stations at optimal points.
    #    c. If still infeasible, try 2-opt swaps to improve time/distance.
    #    d. If still infeasible, return penalty.
    
    # To ensure validity, we can use a "repair" mechanism:
    # If a route is infeasible, we can drop the last customer to the next route (if possible) or move it to a new route.
    
    # Let's implement a GA with:
    # - Chromosome: Permutation of customers.
    # - Decoder: 
    #   1. Iterate through permutation.
    #   2. Maintain current route.
    #   3. Add customer if feasible (demand, battery, time).
    #   4. If not feasible, close current route, start new one.
    #   5. After splitting, run local search (2-opt + station insertion) on each route.
    #   6. If any route is infeasible after local search, apply penalty.
    
    # This is similar to the current program's construction but allows the GA to explore different *orders* and *splits*.
    
    def decode(permutation):
        """
        Decode a permutation into routes.
        Returns (routes, total_distance, is_valid)
        """
        routes = []
        current_route = []
        current_demand = 0
        current_dist = 0.0
        current_time = 0.0
        current_bat = battery_cap
        
        # We need a function to check if adding a customer to the current route is feasible.
        # This is expensive if we re-simulate the whole route.
        # Approximation: Check battery and time loosely?
        # No, we need accuracy.
        
        # Let's use a simpler decoding:
        # 1. Split permutation into chunks of max size such that demand <= capacity.
        # 2. For each chunk, run a local search to make it a valid route.
        # 3. If a chunk fails, split it further.
        
        # This is still complex. Let's use a different approach:
        # **Route-Based GA**
        # Representation: List of routes.
        # Crossover: Take customers from parent 1 in order, assign to routes in parent 2's structure?
        
        # Let's go back to the **Permutation GA** but with a smarter decoder.
        # The decoder will:
        # 1. Iterate through the permutation.
        # 2. Try to insert the customer into the *end* of the current route.
        # 3. If infeasible, try to insert it at the *beginning* of the next route (which is empty)?
        #    No, that's just sequential.
        
        # Let's try a **Memetic Algorithm** (GA + Local Search).
        # The local search is the same as the current program's, but applied to the GA offspring.
        
        # Implementation Plan:
        # 1. Generate initial population of 50 random permutations.
        # 2. Decode each: 
        #    - Split into routes greedily based on capacity.
        #    - For each route, run local search (2-opt, or-opt, station insertion) to minimize distance and ensure feasibility.
        #    - If infeasible, penalty.
        # 3. Evaluate fitness (total distance + penalty).
        # 4. Selection, Crossover (OX), Mutation (swap, insert).
        # 5. Repeat until deadline.
        # 6. Return best valid solution.
        
        # Key: The decoder's local search must be fast.
        
        pass # Placeholder

    # Given the complexity and time limit, a full GA might be too slow if the local search is expensive.
    # The current program's local search is already quite good.
    
    # Let's try a **Different Construction Heuristic**:
    # **Savings Algorithm with Time and Battery Constraints**.
    # 1. Start with each customer in its own route.
    # 2. Calculate savings for merging two routes.
    # 3. Merge if savings > 0 and feasibility is maintained.
    # This is structurally different from "insert one by one".
    
    # Let's implement Clarke-Wright Savings with constraints.
    
    def check_route_feasibility(route):
        """
        Returns (is_feasible, total_distance, total_time, final_battery)
        """
        if not route:
            return True, 0.0, 0.0, battery_cap
            
        total_dist = 0.0
        t = 0.0
        bat = battery_cap
        prev = 0
        
        for node in route:
            d = dist[prev, node]
            total_dist += d
            bat -= d
            if bat < 0:
                return False, total_dist, t, bat
            t += d / speed
            
            if node in stations:
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
            else:
                # Customer
                t = max(t, ready[node])
                if t > due[node]:
                    return False, total_dist, t, bat
                t += service[node]
            
            prev = node
        
        # Return to depot
        d = dist[prev, 0]
        total_dist += d
        bat -= d
        if bat < 0:
            return False, total_dist, t, bat
        t += d / speed
        if t > horizon:
            return False, total_dist, t, bat
            
        return True, total_dist, t, bat

    def get_route_distance(route):
        if not route:
            return 0.0
        d = dist[0, route[0]]
        for i in range(len(route) - 1):
            d += dist[route[i], route[i+1]]
        d += dist[route[-1], 0]
        return d

    def get_route_demand(route):
        return sum(demands[n] for n in route if n not in stations)

    # 1. Initialize: Each customer in its own route.
    #    Check if single customer route is feasible.
    #    If not, we have a problem. But usually singles are feasible.
    routes = []
    for c in range(1, n_customers + 1):
        # Check feasibility of [c]
        feas, _, _, _ = check_route_feasibility([c])
        if feas:
            routes.append([c])
        else:
            # If single is infeasible, we can't serve it? 
            # The task says "Any broken rule makes the whole result invalid".
            # We assume singles are feasible.
            routes.append([c])

    # 2. Calculate savings for all pairs of routes.
    #    Saving = dist(0, a) + dist(0, b) - dist(a, b)
    #    Where a is last of route i, b is first of route j?
    #    Or a is first, b is last?
    #    Standard CW: Route i ends at a, Route j starts at b.
    #    New route: i + j.
    #    Saving = dist(0, a) + dist(0, b) - dist(a, b)
    
    #    We need to consider all possible connections.
    #    Since routes are small initially, we can calculate savings for all pairs.
    
    savings_list = []
    for i in range(len(routes)):
        for j in range(i + 1, len(routes)):
            # Check if routes i and j can be merged.
            # We need to try both orders: i+j and j+i.
            # And also reversing one of them?
            # Standard CW assumes fixed order within routes.
            # But we can reverse.
            
            # Let's try 4 combinations:
            # 1. routes[i] + routes[j]
            # 2. routes[i] + routes[j][::-1]
            # 3. routes[j] + routes[i]
            # 4. routes[j][::-1] + routes[i]
            
            for r1, r2 in [(routes[i], routes[j]), (routes[i], routes[j][::-1]), 
                            (routes[j], routes[i]), (routes[j][::-1], routes[i])]:
                if not r1 or not r2:
                    continue
                a = r1[-1]
                b = r2[0]
                saving = dist[0, a] + dist[0, b] - dist[a, b]
                if saving > 0:
                    savings_list.append((saving, i, j, r1, r2))
                    
    # Sort savings in descending order
    savings_list.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Merge
    #    We need to track which customers are in which route.
    #    Since we are merging, we can maintain a list of active routes.
    #    If a route is merged into another, mark it as inactive.
    
    active_routes = [True] * len(routes)
    
    for saving, i, j, r1, r2 in savings_list:
        if not active_routes[i] or not active_routes[j]:
            continue
            
        # Check if r1 and r2 are still the current content of routes i and j?
        # This is tricky because routes i and j might have changed?
        # No, in CW, we merge two *current* routes.
        # But if we merge i and j, we create a new route.
        # We should replace route i with the merged route and mark j as inactive.
        
        # However, the "saving" was calculated based on the *original* endpoints.
        # If route i or j has been modified by a previous merge, the saving is invalid.
        # So we must re-check feasibility and calculate actual distance change.
        
        # Let's just try to merge routes i and j (current content) in the order that gave the best saving?
        # No, the saving was for specific r1, r2.
        # If routes[i] is not r1, skip.
        
        if routes[i] != r1 or routes[j] != r2:
            continue
            
        merged = r1 + r2
        
        # Check feasibility of merged route
        feas, d_new, t_new, bat_new = check_route_feasibility(merged)
        if not feas:
            continue
            
        # Check demand
        if get_route_demand(merged) > capacity:
            continue
            
        # Calculate distance change
        d_old = get_route_distance(r1) + get_route_distance(r2)
        if d_new < d_old - 1e-9:
            # Merge
            routes[i] = merged
            active_routes[j] = False
            # Note: We don't remove j from routes list to keep indices stable,
            # but we mark it inactive.
            # However, future savings involving j should be skipped.
            
    # 4. Final Local Search
    #    Run 2-opt and Or-opt on all active routes.
    
    def try_2opt(routes):
        improved = False
        for i in range(len(routes)):
            if not active_routes[i]:
                continue
            r = routes[i]
            if len(r) < 2:
                continue
            n = len(r)
            for a in range(n - 1):
                for b in range(a + 1, n):
                    new_r = r[:a] + r[a:b+1][::-1] + r[b+1:]
                    if new_r == r:
                        continue
                    feas, d_new, _, _ = check_route_feasibility(new_r)
                    if not feas:
                        continue
                    d_old = get_route_distance(r)
                    if d_new < d_old - 1e-9:
                        routes[i] = new_r
                        improved = True
        return improved

    def try_oropt(routes):
        improved = False
        for i in range(len(routes)):
            if not active_routes[i]:
                continue
            r = routes[i]
            n = len(r)
            for length in [1, 2, 3]:
                for a in range(n):
                    if a + length > n:
                        continue
                    segment = r[a:a+length]
                    rest = r[:a] + r[a+length:]
                    for b in range(len(rest) + 1):
                        new_r = rest[:b] + segment + rest[b:]
                        if new_r == r:
                            continue
                        feas, d_new, _, _ = check_route_feasibility(new_r)
                        if not feas:
                            continue
                        d_old = get_route_distance(r)
                        if d_new < d_old - 1e-9:
                            routes[i] = new_r
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        return improved

    # Run local search
    while time.time() < deadline:
        imp = try_2opt(routes)
        if imp:
            continue
        imp = try_oropt(routes)
        if imp:
            continue
        break

    # Filter out inactive and empty routes
    final_routes = [routes[i] for i in range(len(routes)) if active_routes[i] and routes[i]]
    
    return final_routes
# EVOLVE-BLOCK-END
