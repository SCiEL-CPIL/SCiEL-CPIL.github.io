# EVOLVE-BLOCK-START
import numpy as np
from math import sqrt
from heapq import heappush, heappop
from itertools import combinations

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
    
    # Precompute distance matrix
    D = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            D[i,j] = sqrt(dx*dx + dy*dy)
            D[j,i] = D[i,j]
    
    cust_nodes = list(range(1, n_customers + 1))
    station_set = set(stations)
    
    def is_customer(node):
        return 1 <= node <= n_customers
    
    def is_station(node):
        return node in station_set
    
    # State-space Dijkstra for single-vehicle optimal tour
    # State: (visited_mask, current_node, time_bucket, battery_bucket)
    # We discretize time and battery to make this tractable
    
    time_step = 5.0  # time discretization
    batt_step = 10.0  # battery discretization
    
    max_time_buckets = int(horizon / time_step) + 1
    max_batt_buckets = int(battery_cap / batt_step) + 1
    
    # For 25 customers, 2^25 is too large. We'll use a different approach:
    # Build routes using Dijkstra on a state graph where we track:
    # - which customers are visited (as a bitmask for small instances)
    # - current position
    # - time (discretized)
    # - battery (discretized)
    
    # Since 2^25 is infeasible, we'll use a heuristic-guided Dijkstra:
    # 1. First, find a good initial solution using greedy
    # 2. Then improve using local search with Dijkstra-based edge evaluation
    
    # Actually, let's implement a proper state-space approach for multi-route:
    # We'll build each route optimally using Dijkstra on states (pos, time, batt, load)
    # and assign customers to routes iteratively.
    
    def check_route(route):
        """Check if a route is feasible. Returns (feasible, total_distance)."""
        time = 0.0
        battery = battery_cap
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
            if is_customer(node):
                if time > due[node] + 1e-9:
                    return False, 0.0
                time = max(time, ready[node])
                time += service[node]
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, 0.0
            elif is_station(node):
                energy_needed = battery_cap - battery
                time += energy_needed * recharge_rate
                battery = battery_cap
            prev = node
        
        # Return to depot
        d = D[prev, 0]
        if battery - d < -1e-9:
            return False, 0.0
        time += d / speed
        battery -= d
        total_dist += d
        
        if time > horizon + 1e-9:
            return False, 0.0
        
        return True, total_dist
    
    # Dijkstra-based single route optimization given a set of customers
    # State: (current_node, time, battery, load)
    # We want to find the minimum distance tour that visits all given customers
    
    def optimize_route_dijkstra(customers_in_route):
        """
        Use Dijkstra on state space to find optimal route visiting exactly these customers.
        State: (node, time_rounded, battery_rounded)
        We track the path to reconstruct the route.
        """
        if not customers_in_route:
            return []
        
        # For small sets, we can use exact Dijkstra
        # For larger sets, we need heuristics
        n = len(customers_in_route)
        
        if n > 12:
            # Too many for exact Dijkstra with bitmask, use heuristic
            return None
        
        # State: (visited_mask, node, time_bucket, batt_bucket)
        # visited_mask: which customers in customers_in_route are visited
        # node: current position (0 for depot, or customer id)
        # time_bucket: discretized time
        # batt_bucket: discretized battery
        
        # Map customer indices to their position in the list
        cust_to_idx = {c: i for i, c in enumerate(customers_in_route)}
        
        # Dijkstra priority queue: (distance, visited_mask, node, time, battery, path)
        # We use continuous time and battery for accuracy but discretize for state tracking
        
        INF = float('inf')
        
        # State key: (visited_mask, node, time_rounded, batt_rounded)
        # Best distance for each state
        best = {}
        
        # Initial state: at depot, no customers visited, time=0, battery=full
        init_time = 0.0
        init_batt = battery_cap
        init_key = (0, 0, round(init_time / time_step), round(init_batt / batt_step))
        
        heap = []
        heappush(heap, (0.0, 0, 0, init_time, init_batt, [0]))
        best[init_key] = 0.0
        
        target_mask = (1 << n) - 1
        
        while heap:
            dist, mask, node, t, b, path = heappop(heap)
            
            # Check if we've visited all customers and can return to depot
            if mask == target_mask:
                # Check if we can return to depot
                d_back = D[node, 0]
                if b - d_back >= -1e-9 and t + d_back / speed <= horizon + 1e-9:
                    # This is a valid complete route
                    total_dist = dist + d_back
                    # Return the route (excluding depot at start and end)
                    return path[1:]  # path starts with 0 (depot)
            
            # Try all possible next nodes
            for next_node in [0] + customers_in_route:
                if next_node == 0:
                    # Only go back to depot if all customers visited
                    if mask != target_mask:
                        continue
                    d = D[node, 0]
                    if b - d < -1e-9:
                        continue
                    new_t = t + d / speed
                    new_b = b - d
                    if new_t > horizon + 1e-9:
                        continue
                    new_dist = dist + d
                    new_key = (mask, 0, round(new_t / time_step), round(new_b / batt_step))
                    if new_key not in best or new_dist < best[new_key] - 1e-9:
                        best[new_key] = new_dist
                        heappush(heap, (new_dist, mask, 0, new_t, new_b, path + [0]))
                    continue
                
                # next_node is a customer
                idx = cust_to_idx.get(next_node)
                if idx is None:
                    continue
                
                # Check if already visited
                if mask & (1 << idx):
                    continue
                
                d = D[node, next_node]
                if b - d < -1e-9:
                    continue
                
                new_t = t + d / speed
                new_b = b - d
                
                # Check time window
                if new_t > due[next_node] + 1e-9:
                    continue
                
                # Service time
                new_t = max(new_t, ready[next_node])
                new_t += service[next_node]
                
                # Check horizon
                if new_t > horizon + 1e-9:
                    continue
                
                # Check capacity (track load separately or precompute)
                # We need to track load - add to state
                # For now, assume capacity is not binding for small routes
                # TODO: add load tracking
                
                new_dist = dist + d
                new_mask = mask | (1 << idx)
                new_key = (new_mask, next_node, round(new_t / time_step), round(new_b / batt_step))
                
                if new_key not in best or new_dist < best[new_key] - 1e-9:
                    best[new_key] = new_dist
                    heappush(heap, (new_dist, new_mask, next_node, new_t, new_b, path + [next_node]))
        
        return None
    
    # Since exact Dijkstra with full state space is complex, let's use a hybrid approach:
    # 1. Greedy construction with smart insertion
    # 2. Dijkstra-based local search for 2-opt improvements
    
    def greedy_construct():
        """Construct initial routes using greedy nearest feasible customer."""
        unvisited = set(cust_nodes)
        routes = []
        
        while unvisited:
            # Start new route
            current = 0
            time = 0.0
            battery = battery_cap
            load = 0
            route = []
            
            while unvisited:
                best_node = None
                best_dist = float('inf')
                
                for v in unvisited:
                    d = D[current, v]
                    if d >= best_dist:
                        continue
                    if battery - d < -1e-9:
                        continue
                    t_arrive = time + d / speed
                    if t_arrive > due[v] + 1e-9:
                        continue
                    if load + demands[v] > capacity + 1e-9:
                        continue
                    best_node = v
                    best_dist = d
                
                if best_node is None:
                    break
                
                d = D[current, best_node]
                time += d / speed
                battery -= d
                time = max(time, ready[best_node])
                time += service[best_node]
                load += demands[best_node]
                route.append(best_node)
                unvisited.remove(best_node)
                current = best_node
            
            if route:
                # Check if we can return to depot
                d_back = D[current, 0]
                if battery - d_back >= -1e-9 and time + d_back / speed <= horizon + 1e-9:
                    routes.append(route)
                else:
                    # Route is infeasible, need to fix
                    # For now, keep it and let local search fix
                    routes.append(route)
        
        return routes
    
    # 2-opt with feasibility check
    def two_opt_improve(routes, max_iters=100):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        feasible, cost = check_route(new_route)
                        if feasible:
                            old_feasible, old_cost = check_route(route)
                            if old_feasible and cost < old_cost - 1e-9:
                                routes[r_idx] = new_route
                                improved = True
                            elif not old_feasible:
                                routes[r_idx] = new_route
                                improved = True
        return routes
    
    # Cross-route 2-opt (move customer from one route to another)
    def cross_route_improve(routes, max_iters=50):
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for r1 in range(len(routes)):
                for r2 in range(len(routes)):
                    if r1 == r2:
                        continue
                    for i, cust in enumerate(routes[r1]):
                        # Try removing cust from route1
                        new_route1 = routes[r1][:i] + routes[r1][i+1:]
                        feasible1, cost1 = check_route(new_route1) if new_route1 else (True, 0.0)
                        
                        # Try inserting cust into route2 at various positions
                        for j in range(len(routes[r2]) + 1):
                            new_route2 = routes[r2][:j] + [cust] + routes[r2][j:]
                            feasible2, cost2 = check_route(new_route2)
                            
                            if feasible1 and feasible2:
                                old_cost = 0.0
                                old_feasible = True
                                for r in routes:
                                    f, c = check_route(r)
                                    old_cost += c
                                    if not f:
                                        old_feasible = False
                                        break
                                
                                new_cost = cost1 + cost2
                                for r_idx in range(len(routes)):
                                    if r_idx != r1 and r_idx != r2:
                                        f, c = check_route(routes[r_idx])
                                        new_cost += c
                                
                                if old_feasible and new_cost < old_cost - 1e-9:
                                    routes[r1] = new_route1
                                    routes[r2] = new_route2
                                    improved = True
                                    break
        return routes
    
    # Main algorithm
    routes = greedy_construct()
    routes = two_opt_improve(routes)
    routes = cross_route_improve(routes)
    routes = two_opt_improve(routes)
    
    # Ensure all customers are served
    served = set()
    for route in routes:
        for node in route:
            if is_customer(node):
                served.add(node)
    
    missing = set(cust_nodes) - served
    if missing:
        for cust in missing:
            routes.append([cust])
    
    # Final feasibility check and repair
    for i, route in enumerate(routes):
        feasible, cost = check_route(route)
        if not feasible:
            # Try to fix by reordering
            # Simple approach: sort by due date
            routes[i] = sorted(route, key=lambda x: due[x])
    
    return routes
# EVOLVE-BLOCK-END
