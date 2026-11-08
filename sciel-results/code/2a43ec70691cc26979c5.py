import numpy as np
import time
import math
import random
import threading

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
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

    customers = list(range(1, n_customers + 1))

    # Pre-compute distance matrix for speed
    n_nodes = len(coords)
    dist = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            dist[i, j] = d
            dist[j, i] = d

    def simulate_route(route):
        """
        Simulate a route. Returns (is_valid, total_distance).
        """
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        for node in route:
            d = dist[cur_node, node]
            cur_time += d / speed
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in stations:
                energy = battery - cur_bat
                cur_time += energy * recharge_rate
                cur_bat = battery
            else:
                load += demands[node]
                if load > capacity + 1e-9:
                    return False, total_dist
                cur_time = max(cur_time, ready[node])
                if cur_time > due[node] + 1e-9:
                    return False, total_dist
                cur_time += service[node]

            cur_node = node

        d = dist[cur_node, 0]
        cur_time += d / speed
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > horizon + 1e-9:
            return False, total_dist

        return True, total_dist

    def compute_total_distance(routes):
        total = 0.0
        for route in routes:
            if not route:
                continue
            cur_node = 0
            for node in route:
                total += dist[cur_node, node]
                cur_node = node
            total += dist[cur_node, 0]
        return total

    def construct_solution(strategy_id, rng):
        """
        Construct a solution using a specific strategy.
        Returns list of routes.
        """
        if strategy_id == 0:
            # Nearest Neighbor with random tie-breaking
            unvisited = set(customers)
            routes = []
            
            # Sort by due date to start with tightest windows
            start_customers = sorted(list(unvisited), key=lambda x: due[x])
            
            for start_c in start_customers:
                if start_c not in unvisited:
                    continue
                
                route = [start_c]
                unvisited.remove(start_c)
                cur_node = start_c
                load = demands[start_c]
                
                # Simple validity check for start
                if not simulate_route(route)[0]:
                    # If start is invalid, we might have to skip or handle, but assume feasible
                    continue

                while unvisited:
                    best_c = None
                    best_cost = float('inf')
                    
                    # Heuristic: prioritize those with tight windows or close by
                    candidates = list(unvisited)
                    # Add some randomness
                    rng.shuffle(candidates)
                    
                    for c in candidates:
                        if load + demands[c] > capacity:
                            continue
                        
                        # Estimate cost: distance from current + distance to depot (if last)
                        # Actually, let's just try insertion into the end or best position
                        # For NN, we just append? No, NN usually appends.
                        # Let's do a simple insertion into the current route to keep it flexible.
                        # But that's expensive. Let's stick to appending for NN speed.
                        
                        new_route = route + [c]
                        is_valid, _ = simulate_route(new_route)
                        if is_valid:
                            cost = dist[cur_node, c] + dist[c, 0] # Rough heuristic
                            if cost < best_cost:
                                best_cost = cost
                                best_c = c
                    
                    if best_c is not None:
                        route.append(best_c)
                        unvisited.remove(best_c)
                        cur_node = best_c
                        load += demands[best_c]
                    else:
                        break
                
                if route:
                    routes.append(route)
            
            return routes

        elif strategy_id == 1:
            # Regret-2 Insertion
            unvisited = set(customers)
            routes = []
            route_loads = []
            
            while unvisited:
                # Calculate regret for each customer
                best_regret = -1
                best_c = None
                best_route_idx = -1
                best_pos = -1
                
                for c in unvisited:
                    costs = []
                    best_cost = float('inf')
                    best_r = -1
                    best_p = -1
                    
                    for r_idx, route in enumerate(routes):
                        if route_loads[r_idx] + demands[c] > capacity:
                            continue
                        
                        for pos in range(len(route) + 1):
                            new_route = route[:pos] + [c] + route[pos:]
                            is_valid, total_d = simulate_route(new_route)
                            if is_valid:
                                # Cost is increase in distance
                                # Approx: dist(prev, c) + dist(c, next) - dist(prev, next)
                                # But we have total_d. Let's use total_d - old_total?
                                # We don't have old_total stored.
                                # For simplicity in regret, just use total_d as proxy for "quality"
                                # Lower total_d is better.
                                if total_d < best_cost:
                                    best_cost = total_d
                                    best_r = r_idx
                                    best_p = pos
                                    costs.append(total_d)
                    
                    if not costs:
                        continue
                    
                    # Sort costs to get regret = second_best - best
                    costs.sort()
                    if len(costs) > 1:
                        regret = costs[1] - costs[0]
                    else:
                        regret = 0.0
                    
                    if regret > best_regret:
                        best_regret = regret
                        best_c = c
                        best_route_idx = best_r
                        best_pos = best_p
                
                if best_c is None:
                    # No insertion possible, start new route
                    # Pick customer with tightest due date
                    best_c = min(unvisited, key=lambda x: due[x])
                    new_route = [best_c]
                    is_valid, _ = simulate_route(new_route)
                    if is_valid:
                        routes.append(new_route)
                        route_loads.append(demands[best_c])
                        unvisited.remove(best_c)
                    else:
                        unvisited.remove(best_c) # Should not happen
                else:
                    route = routes[best_route_idx]
                    new_route = route[:best_pos] + [best_c] + route[best_pos:]
                    routes[best_route_idx] = new_route
                    route_loads[best_route_idx] += demands[best_c]
                    unvisited.remove(best_c)
            
            return routes

        else:
            # Random Insertion
            unvisited = set(customers)
            routes = []
            route_loads = []
            
            while unvisited:
                c = rng.choice(list(unvisited))
                best_cost = float('inf')
                best_route_idx = -1
                best_pos = -1
                
                for r_idx, route in enumerate(routes):
                    if route_loads[r_idx] + demands[c] > capacity:
                        continue
                    for pos in range(len(route) + 1):
                        new_route = route[:pos] + [c] + route[pos:]
                        is_valid, total_d = simulate_route(new_route)
                        if is_valid and total_d < best_cost:
                            best_cost = total_d
                            best_route_idx = r_idx
                            best_pos = pos
                
                if best_route_idx == -1:
                    new_route = [c]
                    is_valid, _ = simulate_route(new_route)
                    if is_valid:
                        routes.append(new_route)
                        route_loads.append(demands[c])
                    else:
                        continue
                else:
                    route = routes[best_route_idx]
                    new_route = route[:best_pos] + [c] + route[best_pos:]
                    routes[best_route_idx] = new_route
                    route_loads[best_route_idx] += demands[c]
                
                unvisited.remove(c)
            
            return routes

    def local_search(routes, time_limit):
        """
        Improve routes using 2-opt and Or-opt.
        """
        current_total_dist = compute_total_distance(routes)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() - start_ls < time_limit:
            improved = False
            
            # 2-opt within routes
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid:
                            # Check if total distance improves
                            # We need to compare against old route dist
                            old_dist = 0
                            cur_node = 0
                            for node in route:
                                old_dist += dist[cur_node, node]
                                cur_node = node
                            old_dist += dist[cur_node, 0]
                            
                            if new_dist < old_dist - 1e-6:
                                routes[r_idx] = new_route
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if time.time() - start_ls >= time_limit:
                break
                
            # Or-opt (move 1, 2, or 3 nodes) within routes
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                
                for seg_len in [1, 2, 3]:
                    for i in range(n - seg_len + 1):
                        seg = route[i:i+seg_len]
                        rest = route[:i] + route[i+seg_len:]
                        
                        # Try inserting seg into rest
                        for j in range(len(rest) + 1):
                            new_route = rest[:j] + seg + rest[j:]
                            # Avoid same position
                            if new_route == route:
                                continue
                            
                            is_valid, new_dist = simulate_route(new_route)
                            if is_valid:
                                old_dist = 0
                                cur_node = 0
                                for node in route:
                                    old_dist += dist[cur_node, node]
                                    cur_node = node
                                old_dist += dist[cur_node, 0]
                                
                                if new_dist < old_dist - 1e-6:
                                    routes[r_idx] = new_route
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if time.time() - start_ls >= time_limit:
                break

        return routes

    def solve_thread(seed, result_container, lock):
        rng = random.Random(seed)
        strategy = rng.choice([0, 1, 2])
        
        try:
            routes = construct_solution(strategy, rng)
            
            # Validate
            valid = True
            for route in routes:
                if not simulate_route(route)[0]:
                    valid = False
                    break
            
            if not valid:
                return
            
            # Local Search
            # Allocate remaining time to LS
            elapsed = time.time() - start_time
            remaining = deadline - elapsed - 2.0 # 2s buffer
            if remaining > 1.0:
                routes = local_search(routes, remaining)
            
            # Final validation
            valid = True
            served = set()
            for route in routes:
                if not simulate_route(route)[0]:
                    valid = False
                    break
                for node in route:
                    if node in stations:
                        continue
                    if node in served:
                        valid = False
                        break
                    served.add(node)
            
            if not valid or served != set(customers):
                return
            
            total_dist = compute_total_distance(routes)
            
            with lock:
                if result_container[0] is None or total_dist < result_container[0][1]:
                    result_container[0] = (routes, total_dist)
        except Exception:
            pass

    result_container = [None]
    lock = threading.Lock()
    
    # Launch multiple threads
    n_threads = 8
    threads = []
    
    for i in range(n_threads):
        t = threading.Thread(target=solve_thread, args=(i, result_container, lock))
        t.start()
        threads.append(t)
    
    # Wait for all threads to finish or deadline
    for t in threads:
        t.join()
    
    if result_container[0] is not None:
        return result_container[0][0]
    
    # Fallback: simple greedy
    # This should rarely happen
    routes = []
    unvisited = set(customers)
    for c in sorted(customers, key=lambda x: due[x]):
        routes.append([c])
    
    return routes

# EVOLVE-BLOCK-END
