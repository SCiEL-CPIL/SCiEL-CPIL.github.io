import numpy as np
import time
import math
import random
from scipy.spatial.distance import cdist

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Extract parameters
    coords = np.array(instance["coords"], dtype=float)
    demands = instance["demands"]
    ready = instance["ready"]
    due = instance["due"]
    service = instance["service"]
    n_customers = instance["n_customers"]
    stations = set(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]
    
    n_nodes = len(coords)
    
    # Precompute distance matrix using scipy for speed
    dist = cdist(coords, coords)
    
    # Customer IDs are 1 to n_customers
    customers = list(range(1, n_customers + 1))
    
    def decode_and_evaluate(perm):
        """
        Decodes a permutation of customers into feasible routes.
        Returns (total_distance, routes, is_valid).
        
        Strategy:
        1. Iterate through the permutation.
        2. Build a current route.
        3. Check if adding the next customer is feasible (capacity, battery, time).
        4. If not feasible, close the current route and start a new one.
        5. During route building, automatically insert charging stations if battery is low.
        6. If a customer cannot be served in the current route due to time/battery even with charging, 
           close the route and try to start a new one. If it still fails, the solution is invalid.
        """
        routes = []
        current_route = []
        current_demand = 0
        current_dist = 0.0
        current_time = 0.0
        current_battery = battery_cap
        last_node = 0
        
        # We need to simulate the route carefully.
        # A route is a sequence of customers. Stations are inserted dynamically.
        # However, the output format requires a list of node IDs.
        # The decoder should return the *actual* sequence of nodes including stations.
        
        # Let's refine the decoding:
        # We build the customer sequence first, then expand it with stations?
        # No, battery depends on the path. We must decide stations on the fly.
        
        # State: last_node, time, battery
        # For each customer in perm:
        #   Try to go from last_node to customer.
        #   If battery < dist(last_node, customer), we must charge.
        #   Where to charge? The closest station to the path? Or just the closest station to last_node?
        #   Actually, we can treat stations as "virtual nodes" that can be visited.
        #   A simpler heuristic: If battery is insufficient, go to the nearest station that allows reaching the next customer, then go to the customer.
        #   If time is an issue, we might need to charge earlier.
        
        # Let's try a robust decoder:
        # 1. Start a new route.
        # 2. For each customer in the current segment of the permutation:
        #    a. Check capacity. If exceeded, close route.
        #    b. Determine path from last_node to customer.
        #       - Direct? If battery >= dist and time_ok, go direct.
        #       - If not, find a station to visit.
        #         Heuristic: Pick the station 's' that minimizes dist(last, s) + dist(s, customer) 
        #         subject to battery(last->s) <= current_battery and battery(s->customer) <= battery_cap.
        #         If no such station, route is infeasible for this customer.
        #    c. Update time, battery, last_node.
        # 3. Close route at the end of the permutation or when a customer doesn't fit.
        
        # This is complex to implement efficiently inside a GA loop.
        # Alternative: The GA permutes customers. The decoder assigns them to routes 
        # by splitting the permutation into chunks that are feasible.
        
        # Let's use a simpler "First Fit Decreasing" style decoding on the permutation?
        # No, the permutation defines the order.
        
        # Revised Decoder Logic:
        # The input is a permutation of all customers.
        # We scan left to right. We maintain a current route.
        # For each customer c in the permutation:
        #   1. Can we add c to the current route?
        #      - Check capacity.
        #      - Check if there exists a feasible path from the end of the current route to c, 
        #        possibly via stations, that ends at c and allows return to depot (for the final customer).
        #      - Actually, we don't need to return to depot until the route is closed.
        #      - But we must ensure battery doesn't drop below 0.
        
        # Let's implement a helper: can_insert(c, last_node, time, battery, route_customers)
        # This is expensive.
        
        # Better approach for GA: 
        # Use a "Sweep" or "Angle" based initial population?
        # Or just random permutations.
        
        # Let's try a different representation: 
        # The chromosome is a list of (route_id, customer_id).
        # But standard GA on permutations is easier.
        
        # Let's stick to: Permutation of customers.
        # Decoder: 
        #   routes = []
        #   current = []
        #   for c in perm:
        #      if not feasible_to_add(c, current):
        #          if current: routes.append(finalize(current))
        #          current = [c]
        #      else:
        #          current.append(c)
        #   if current: routes.append(finalize(current))
        
        # feasible_to_add(c, current):
        #   Simulate the route 'current' + [c].
        #   Check capacity.
        #   Check battery/time with optimal station insertion.
        #   This simulation is O(N_stations * len(current)).
        
        # To make it fast, we can approximate station insertion:
        # If battery < dist(last, c), we MUST stop at a station.
        # We pick the station s that minimizes detour.
        
        def simulate_route(customers_seq):
            """
            Simulates a route of customers. Returns (distance, time, valid).
            Automatically inserts stations if needed.
            """
            if not customers_seq:
                return 0.0, 0.0, True
            
            total_dist = 0.0
            t = 0.0
            bat = battery_cap
            last = 0
            
            for c in customers_seq:
                # Check capacity is done outside, but let's assume it's fine.
                
                # Determine if we need to charge
                d_direct = dist[last, c]
                
                if bat < d_direct:
                    # Need to go to a station
                    # Find best station s
                    best_s = -1
                    best_cost = float('inf')
                    
                    for s in stations:
                        d1 = dist[last, s]
                        d2 = dist[s, c]
                        # Check battery for last->s
                        if bat < d1:
                            continue
                        # Check battery for s->c (after full charge)
                        if battery_cap < d2:
                            continue
                        
                        # Cost is extra distance: d1 + d2 - d_direct
                        cost = d1 + d2 - d_direct
                        if cost < best_cost:
                            best_cost = cost
                            best_s = s
                            
                    if best_s == -1:
                        return 0.0, 0.0, False # Cannot reach customer even with charging
                    
                    # Go to station
                    d1 = dist[last, best_s]
                    total_dist += d1
                    bat -= d1
                    t += d1 / speed
                    
                    # Charge
                    charge_time = (battery_cap - bat) * recharge_rate
                    t += charge_time
                    bat = battery_cap
                    last = best_s
                    
                    # Go to customer
                    d2 = dist[last, c]
                    total_dist += d2
                    bat -= d2
                    t += d2 / speed
                    last = c
                    
                else:
                    # Direct
                    total_dist += d_direct
                    bat -= d_direct
                    t += d_direct / speed
                    last = c
                
                # At customer c
                t_arrival = t
                t_start = max(t_arrival, ready[c])
                if t_start > due[c]:
                    return 0.0, 0.0, False
                t = t_start + service[c]
            
            # Return to depot
            d_back = dist[last, 0]
            if bat < d_back:
                # Need to charge before returning?
                # Find station s such that last->s->depot is feasible
                best_s = -1
                best_cost = float('inf')
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, 0]
                    if bat < d1:
                        continue
                    if battery_cap < d2:
                        continue
                    cost = d1 + d2 - d_back
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                
                if best_s == -1:
                    return 0.0, 0.0, False
                
                d1 = dist[last, best_s]
                total_dist += d1
                bat -= d1
                t += d1 / speed
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                last = best_s
                
                d2 = dist[last, 0]
                total_dist += d2
                bat -= d2
                t += d2 / speed
            else:
                total_dist += d_back
                bat -= d_back
                t += d_back / speed
            
            if t > horizon:
                return 0.0, 0.0, False
            
            return total_dist, t, True

        # Decoder
        routes = []
        current_route = []
        current_demand = 0
        
        for c in perm:
            # Check capacity
            if current_demand + demands[c] > capacity:
                if current_route:
                    d, t, valid = simulate_route(current_route)
                    if valid:
                        routes.append(current_route)
                    else:
                        return 0.0, [], False # Invalid solution
                    current_route = []
                    current_demand = 0
            
            # Try to add c
            new_route = current_route + [c]
            d, t, valid = simulate_route(new_route)
            
            if valid:
                current_route = new_route
                current_demand += demands[c]
            else:
                # Cannot add to current route. Close current, start new.
                if current_route:
                    d, t, valid_prev = simulate_route(current_route)
                    if valid_prev:
                        routes.append(current_route)
                    else:
                        return 0.0, [], False
                    current_route = []
                    current_demand = 0
                
                # Start new route with c
                d, t, valid_single = simulate_route([c])
                if not valid_single:
                    return 0.0, [], False
                current_route = [c]
                current_demand = demands[c]
        
        if current_route:
            d, t, valid = simulate_route(current_route)
            if valid:
                routes.append(current_route)
            else:
                return 0.0, [], False
        
        total_dist = sum(simulate_route(r)[0] for r in routes)
        return total_dist, routes, True

    # Initial Population
    pop_size = 50
    population = []
    for _ in range(pop_size):
        random.shuffle(customers)
        population.append(customers[:])
    
    # Evaluate initial population
    best_perm = None
    best_dist = float('inf')
    best_routes = None
    
    for p in population:
        d, r, valid = decode_and_evaluate(p)
        if valid and d < best_dist:
            best_dist = d
            best_perm = p[:]
            best_routes = r
    
    if best_perm is None:
        # Fallback: single customer per route
        return [[c] for c in customers]
        
    # GA Loop
    # We need to expand the routes back to node sequences for the final output?
    # No, the task says "Return a list of routes, each a list of node ids".
    # The decoder returns customer lists. We need to convert them to node lists including stations.
    # Wait, the simulate_route function doesn't return the *path* with stations, just the distance.
    # We need a function that returns the actual sequence of nodes.
    
    def get_route_nodes(customers_seq):
        """Returns the list of node IDs (including stations) for a feasible customer sequence."""
        if not customers_seq:
            return []
        
        nodes = []
        t = 0.0
        bat = battery_cap
        last = 0
        
        for c in customers_seq:
            d_direct = dist[last, c]
            
            if bat < d_direct:
                # Find best station
                best_s = -1
                best_cost = float('inf')
                for s in stations:
                    d1 = dist[last, s]
                    d2 = dist[s, c]
                    if bat < d1 or battery_cap < d2:
                        continue
                    cost = d1 + d2 - d_direct
                    if cost < best_cost:
                        best_cost = cost
                        best_s = s
                if best_s == -1:
                    return None # Should not happen if simulate_route passed
                nodes.append(best_s)
                d1 = dist[last, best_s]
                bat -= d1
                t += d1 / speed
                charge_time = (battery_cap - bat) * recharge_rate
                t += charge_time
                bat = battery_cap
                last = best_s
                d2 = dist[last, c]
                bat -= d2
                t += d2 / speed
                last = c
            else:
                d_direct = dist[last, c]
                bat -= d_direct
                t += d_direct / speed
                last = c
            
            nodes.append(c)
            t_arrival = t
            t_start = max(t_arrival, ready[c])
            t = t_start + service[c]
        
        # Return to depot
        d_back = dist[last, 0]
        if bat < d_back:
            best_s = -1
            best_cost = float('inf')
            for s in stations:
                d1 = dist[last, s]
                d2 = dist[s, 0]
                if bat < d1 or battery_cap < d2:
                    continue
                cost = d1 + d2 - d_back
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
            if best_s == -1:
                return None
            nodes.append(best_s)
            d1 = dist[last, best_s]
            bat -= d1
            t += d1 / speed
            charge_time = (battery_cap - bat) * recharge_rate
            t += charge_time
            bat = battery_cap
            last = best_s
            d2 = dist[last, 0]
            bat -= d2
            t += d2 / speed
        else:
            d_back = dist[last, 0]
            bat -= d_back
            t += d_back / speed
        
        return nodes

    # Convert best_routes to node lists
    final_routes = []
    for r in best_routes:
        nodes = get_route_nodes(r)
        if nodes:
            final_routes.append(nodes)
        else:
            # Fallback
            final_routes.append(r)
            
    # Local Search on the best solution?
    # We can apply 2-opt on the customer sequences before decoding.
    
    def local_search(perm):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # 2-opt
            for i in range(len(perm) - 1):
                for j in range(i + 1, len(perm)):
                    new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                    d, r, valid = decode_and_evaluate(new_perm)
                    if valid:
                        old_d, _, _ = decode_and_evaluate(perm)
                        if d < old_d - 1e-9:
                            perm = new_perm
                            improved = True
                            break
                if improved:
                    break
        return perm

    # Apply local search to the best individual
    # Note: decode_and_evaluate is expensive. We should cache or optimize.
    # For now, let's just run a few generations of GA.
    
    generations = 50
    for gen in range(generations):
        if time.time() > deadline - 10:
            break
            
        new_population = [best_perm[:]]
        
        # Crossover and Mutation
        while len(new_population) < pop_size:
            # Select parents (Tournament)
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Order Crossover
            a, b = sorted(random.sample(range(len(p1)), 2))
            child = [None] * len(p1)
            child[a:b+1] = p1[a:b+1]
            fill = [x for x in p2 if x not in child[a:b+1]]
            idx = 0
            for i in range(len(child)):
                if child[i] is None:
                    child[i] = fill[idx]
                    idx += 1
            
            # Mutation
            if random.random() < 0.2:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]
            
            new_population.append(child)
        
        # Evaluate new population
        for p in new_population:
            d, r, valid = decode_and_evaluate(p)
            if valid and d < best_dist:
                best_dist = d
                best_perm = p[:]
                best_routes = r
        
        population = new_population
        
        # Apply local search to the global best occasionally
        if gen % 10 == 0:
            best_perm = local_search(best_perm)
            d, r, valid = decode_and_evaluate(best_perm)
            if valid:
                best_dist = d
                best_routes = r

    # Final conversion
    final_routes = []
    for r in best_routes:
        nodes = get_route_nodes(r)
        if nodes:
            final_routes.append(nodes)
        else:
            final_routes.append(r)
            
    return final_routes
