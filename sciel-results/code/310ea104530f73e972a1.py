import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    deadline = start_time + 145.0

    # Extract instance data
    coords = np.array(instance["coords"], dtype=np.float64)
    n_customers = instance["n_customers"]
    demands = np.array(instance["demands"], dtype=np.float64)
    ready = np.array(instance["ready"], dtype=np.float64)
    due = np.array(instance["due"], dtype=np.float64)
    service = np.array(instance["service"], dtype=np.float64)
    stations_set = set(instance["stations"])
    capacity = instance["capacity"]
    battery = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    customers = list(range(1, n_customers + 1))
    n_nodes = len(coords)
    
    # Pre-compute full distance matrix
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=2))

    # Helper to check feasibility and get distance.
    def route_distance_and_feasibility(route):
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
            
            if node in stations_set:
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

    # --- FAST CONSTRUCTION: Nearest Neighbor ---
    def construct_solution(seed_offset):
        rng = random.Random(42 + seed_offset)
        unvisited = set(customers)
        routes = []
        
        while unvisited:
            route = []
            cur_node = 0
            cur_time = 0.0
            cur_bat = battery
            load = 0.0
            
            while unvisited:
                best_c = None
                best_score = float('inf')
                
                feasible_nexts = []
                for c in unvisited:
                    d = dist[cur_node, c]
                    new_time = cur_time + d / speed
                    new_bat = cur_bat - d
                    if new_bat < -1e-9:
                        continue
                    if load + demands[c] > capacity:
                        continue
                    new_time = max(new_time, ready[c])
                    if new_time > due[c]:
                        continue
                    new_time += service[c]
                    feasible_nexts.append((c, new_time, new_bat, d))
                
                if not feasible_nexts:
                    break
                
                for c, nt, nb, d in feasible_nexts:
                    score = d + rng.random() * 5
                    if score < best_score:
                        best_score = score
                        best_c = c
                        best_nt = nt
                        best_nb = nb
                        best_d = d
                
                if best_c is None:
                    break
                    
                route.append(best_c)
                cur_node = best_c
                cur_time = best_nt
                cur_bat = best_nb
                load += demands[best_c]
                unvisited.remove(best_c)
                
            if route:
                d_back = dist[cur_node, 0]
                if cur_bat - d_back >= -1e-9:
                    t_back = cur_time + d_back / speed
                    if t_back <= horizon:
                        routes.append(route)
                    else:
                        while route and t_back > horizon:
                            last = route.pop()
                            unvisited.add(last)
                            cur_node = route[-1] if route else 0
                            cur_time = 0.0
                            cur_bat = battery
                            load = 0.0
                            for node in route:
                                d = dist[cur_node, node]
                                cur_time += d / speed
                                cur_bat -= d
                                if node not in stations_set:
                                    load += demands[node]
                                    cur_time = max(cur_time, ready[node])
                                    cur_time += service[node]
                                cur_node = node
                            d_back = dist[cur_node, 0]
                            t_back = cur_time + d_back / speed
                            if not route:
                                break
                        if route and cur_bat - d_back >= -1e-9 and t_back <= horizon:
                            routes.append(route)
                
                if not route:
                    break
        
        # Fallback for any remaining unvisited customers
        if unvisited:
            for c in list(unvisited):
                inserted = False
                for idx, r in enumerate(routes):
                    for pos in range(len(r) + 1):
                        new_r = r[:pos] + [c] + r[pos:]
                        if route_distance_and_feasibility(new_r)[0]:
                            routes[idx] = new_r
                            unvisited.remove(c)
                            inserted = True
                            break
                    if inserted:
                        break
                if not inserted:
                    if route_distance_and_feasibility([c])[0]:
                        routes.append([c])
                        unvisited.remove(c)
                    else:
                        routes.append([c])
                        unvisited.remove(c)
        
        return routes

    # Generate initial solution
    best_routes = None
    best_dist = float('inf')
    
    for i in range(20):
        if time.time() > deadline - 10:
            break
        routes = construct_solution(i)
        if routes:
            d = compute_total_distance(routes)
            valid = True
            served = set()
            for route in routes:
                if not route_distance_and_feasibility(route)[0]:
                    valid = False
                    break
                for node in route:
                    if node not in stations_set:
                        if node in served:
                            valid = False
                            break
                        served.add(node)
            
            if valid and served == set(customers):
                if d < best_dist:
                    best_dist = d
                    best_routes = [list(r) for r in routes]
    
    if best_routes is None:
        return [[c] for c in customers]

    # --- PARALLEL TEMPERING SIMULATED ANNEALING ---
    def pt_simulated_annealing(initial_routes, time_limit):
        # Initialize chains
        n_chains = 5
        chains = []
        chain_costs = []
        best_overall = [list(r) for r in initial_routes]
        best_overall_cost = compute_total_distance(best_overall)
        
        # Create diverse initial chains
        for i in range(n_chains):
            if i == 0:
                chains.append([list(r) for r in initial_routes])
            else:
                # Perturb initial solution
                perturbed = [list(r) for r in initial_routes]
                rng = random.Random(100 + i)
                for _ in range(5):
                    if len(perturbed) > 1:
                        r1 = rng.randint(0, len(perturbed) - 1)
                        if perturbed[r1]:
                            idx = rng.randint(0, len(perturbed[r1]) - 1)
                            node = perturbed[r1].pop(idx)
                            r2 = rng.randint(0, len(perturbed) - 1)
                            if r2 != r1:
                                pos = rng.randint(0, len(perturbed[r2]))
                                new_r2 = perturbed[r2][:pos] + [node] + perturbed[r2][pos:]
                                new_r1 = perturbed[r1]
                                if route_distance_and_feasibility(new_r1)[0] and route_distance_and_feasibility(new_r2)[0]:
                                    perturbed[r1] = new_r1
                                    perturbed[r2] = new_r2
                chains.append(perturbed)
            
            cost = compute_total_distance(chains[i])
            chain_costs.append(cost)
            if cost < best_overall_cost:
                best_overall_cost = cost
                best_overall = [list(r) for r in chains[i]]
        
        # Temperature schedule
        t_start = 50.0
        t_end = 0.1
        t_current = t_start
        cooling_rate = 0.9995
        
        start_pt = time.time()
        rng = random.Random(42)
        
        while time.time() - start_pt < time_limit:
            for chain_idx in range(n_chains):
                if time.time() > deadline - 2:
                    break
                
                current = chains[chain_idx]
                current_cost = chain_costs[chain_idx]
                
                # Generate a neighbor move
                move_type = rng.random()
                new_routes = [list(r) for r in current]
                valid_move = False
                
                if len(new_routes) > 0:
                    if move_type < 0.4:
                        # Relocate
                        r1 = rng.randint(0, len(new_routes) - 1)
                        if new_routes[r1]:
                            idx = rng.randint(0, len(new_routes[r1]) - 1)
                            node = new_routes[r1].pop(idx)
                            r2 = rng.randint(0, len(new_routes) - 1)
                            if r2 != r1:
                                pos = rng.randint(0, len(new_routes[r2]))
                                new_r2 = new_routes[r2][:pos] + [node] + new_routes[r2][pos:]
                                if route_distance_and_feasibility(new_routes[r1])[0] and route_distance_and_feasibility(new_r2)[0]:
                                    new_routes[r2] = new_r2
                                    valid_move = True
                            else:
                                # Reinsert in same route
                                pos = rng.randint(0, len(new_routes[r1]))
                                new_r1 = new_routes[r1][:pos] + [node] + new_routes[r1][pos:]
                                if route_distance_and_feasibility(new_r1)[0]:
                                    new_routes[r1] = new_r1
                                    valid_move = True
                    elif move_type < 0.7:
                        # Swap
                        if len(new_routes) > 1:
                            r1 = rng.randint(0, len(new_routes) - 1)
                            r2 = rng.randint(0, len(new_routes) - 1)
                            if r1 != r2 and new_routes[r1] and new_routes[r2]:
                                i = rng.randint(0, len(new_routes[r1]) - 1)
                                j = rng.randint(0, len(new_routes[r2]) - 1)
                                n1, n2 = new_routes[r1][i], new_routes[r2][j]
                                new_r1 = new_routes[r1][:i] + [n2] + new_routes[r1][i+1:]
                                new_r2 = new_routes[r2][:j] + [n1] + new_routes[r2][j+1:]
                                if route_distance_and_feasibility(new_r1)[0] and route_distance_and_feasibility(new_r2)[0]:
                                    new_routes[r1] = new_r1
                                    new_routes[r2] = new_r2
                                    valid_move = True
                    else:
                        # 2-opt within a route
                        r_idx = rng.randint(0, len(new_routes) - 1)
                        route = new_routes[r_idx]
                        if len(route) >= 2:
                            i = rng.randint(0, len(route) - 2)
                            j = rng.randint(i + 1, len(route) - 1)
                            new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                            if route_distance_and_feasibility(new_route)[0]:
                                new_routes[r_idx] = new_route
                                valid_move = True
                
                if valid_move:
                    new_cost = compute_total_distance(new_routes)
                    delta = new_cost - current_cost
                    
                    # Acceptance probability
                    if delta < 0 or rng.random() < math.exp(-delta / max(t_current, 1e-9)):
                        chains[chain_idx] = new_routes
                        chain_costs[chain_idx] = new_cost
                        
                        if new_cost < best_overall_cost:
                            best_overall_cost = new_cost
                            best_overall = [list(r) for r in new_routes]
                    
                    # Parallel tempering: swap with a hotter chain
                    if chain_idx < n_chains - 1:
                        swap_idx = chain_idx + 1
                        # Simple swap based on cost
                        if chain_costs[chain_idx] > chain_costs[swap_idx] and rng.random() < 0.5:
                            chains[chain_idx], chains[swap_idx] = chains[swap_idx], chains[chain_idx]
                            chain_costs[chain_idx], chain_costs[swap_idx] = chain_costs[swap_idx], chain_costs[chain_idx]
                
                # Cool down
                t_current *= cooling_rate
                if t_current < t_end:
                    t_current = t_end
        
        return best_overall

    # Run parallel tempering SA
    elapsed = time.time() - start_time
    remaining = deadline - elapsed - 2.0
    if remaining > 1.0:
        best_routes = pt_simulated_annealing(best_routes, remaining)
        
    # Final validation
    valid = True
    served = set()
    for route in best_routes:
        if not route_distance_and_feasibility(route)[0]:
            valid = False
            break
        for node in route:
            if node not in stations_set:
                if node in served:
                    valid = False
                    break
                served.add(node)
    
    if not valid or served != set(customers):
        # Fallback to individual routes
        best_routes = [[c] for c in customers]
        
    return best_routes
