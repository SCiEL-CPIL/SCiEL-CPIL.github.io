import numpy as np
import time
import math
import random

def plan(instance):
    start_time = time.time()
    # 150s limit, keep 5s safety margin
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
    
    # Pre-compute distance matrix using NumPy for speed
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    def simulate_route(route):
        """
        Simulate a route. Returns (is_valid, total_distance).
        Optimized for speed: local variable references, minimal overhead.
        """
        if not route:
            return True, 0.0

        cur_node = 0
        cur_time = 0.0
        cur_bat = battery
        total_dist = 0.0
        load = 0.0

        # Local references for speed
        d_mat = dist
        req = demands
        rd = ready
        dd = due
        sv = service
        st = stations_set
        cap = capacity
        bat_cap = battery
        spd = speed
        rch = recharge_rate
        hor = horizon

        for node in route:
            d = d_mat[cur_node, node]
            cur_time += d / spd
            cur_bat -= d
            total_dist += d

            if cur_bat < -1e-9:
                return False, total_dist

            if node in st:
                energy = bat_cap - cur_bat
                cur_time += energy * rch
                cur_bat = bat_cap
            else:
                load += req[node]
                if load > cap + 1e-9:
                    return False, total_dist
                if cur_time < rd[node]:
                    cur_time = rd[node]
                if cur_time > dd[node] + 1e-9:
                    return False, total_dist
                cur_time += sv[node]

            cur_node = node

        d = d_mat[cur_node, 0]
        cur_time += d / spd
        cur_bat -= d
        total_dist += d

        if cur_bat < -1e-9:
            return False, total_dist
        if cur_time > hor + 1e-9:
            return False, total_dist

        return True, total_dist

    def compute_total_distance(routes):
        total = 0.0
        d_mat = dist
        for route in routes:
            if not route:
                continue
            cur_node = 0
            for node in route:
                total += d_mat[cur_node, node]
                cur_node = node
            total += d_mat[cur_node, 0]
        return total

    def construct_swept_cluster_solution(rng):
        """
        Strategy: Swept-Clustering Construction.
        1. Calculate angles of all customers relative to the depot.
        2. Sort customers by angle.
        3. Determine cluster boundaries by sweeping a line from the depot,
           grouping customers into contiguous angular sectors.
        4. The number of clusters is determined by capacity and battery constraints.
        5. Within each cluster, use a fast nearest-neighbor or simple insertion
           to build the route.
        
        This creates geographically coherent routes, which are often better 
        starting points for local search than pure NN which can get stuck 
        in local minima due to myopic decisions.
        """
        # 1. Compute angles
        angles = np.arctan2(coords[1:, 1], coords[1:, 0]) # coords[1:] are customers
        # Map customer id to angle index
        # customers are 1..n_customers, coords[1:] correspond to them
        
        # 2. Sort by angle
        sorted_indices = np.argsort(angles)
        sorted_customers = [customers[i] for i in sorted_indices]
        
        # 3. Determine number of clusters
        # Heuristic: Estimate total demand and divide by capacity.
        total_demand = sum(demands[c] for c in customers)
        min_clusters_by_cap = math.ceil(total_demand / capacity)
        
        # Also consider battery. If total distance is large, we need more clusters.
        # Rough estimate of total tour length: 2 * sum(dist(0, c)) * 0.7 (savings factor)
        total_dist_est = 2 * sum(dist[0, c] for c in customers) * 0.7
        min_clusters_by_bat = math.ceil(total_dist_est / battery)
        
        # Add a safety margin. We want clusters to be tight.
        # Let's aim for slightly more clusters than the minimum to allow for 
        # better routing within them.
        num_clusters = max(min_clusters_by_cap, min_clusters_by_bat)
        
        # Randomize the number of clusters slightly to create diversity
        # num_clusters = max(1, num_clusters + rng.randint(-1, 1))
        
        # 4. Assign customers to clusters
        # We want contiguous blocks in the sorted-by-angle list.
        # To ensure balance, we can try to distribute them evenly by index,
        # but also respect the "swept" nature.
        
        # A simple swept approach:
        # Start at angle 0, sweep 360.
        # Accumulate customers until capacity or battery is nearly full.
        # Close the cluster, start a new one.
        
        # However, simple sweeping might create unbalanced clusters.
        # Let's use a "balanced swept" approach:
        # Divide the sorted list into `num_clusters` roughly equal segments.
        
        # But wait, "swept" usually implies dynamic boundaries based on constraints.
        # Let's stick to a hybrid: 
        # We know we need at least `num_clusters`. 
        # Let's just split the sorted list into `num_clusters` chunks.
        # This ensures geographic coherence (each chunk is a sector).
        
        chunks = []
        chunk_size = n_customers // num_clusters
        remainder = n_customers % num_clusters
        
        start_idx = 0
        for i in range(num_clusters):
            size = chunk_size + (1 if i < remainder else 0)
            if size == 0:
                continue
            chunk = sorted_customers[start_idx : start_idx + size]
            chunks.append(chunk)
            start_idx += size
        
        # 5. Build routes for each chunk
        routes = []
        for chunk in chunks:
            if not chunk:
                continue
            
            # Use a simple nearest-neighbor within the chunk to build the sequence
            # Starting from the depot.
            # Since the chunk is a sector, NN should work well.
            
            remaining = set(chunk)
            route = []
            cur_node = 0
            cur_time = 0.0
            cur_bat = battery
            cur_load = 0
            
            while remaining:
                best_c = -1
                best_d = float('inf')
                
                for c in remaining:
                    d = dist[cur_node, c]
                    if d < best_d:
                        # Feasibility check
                        if cur_load + demands[c] > capacity:
                            continue
                        if cur_bat - d < 0:
                            # If battery is low, we might need to charge.
                            # For simplicity in construction, if we can't reach, 
                            # we might have to end the route or charge.
                            # Let's check if we can return to depot and recharge? 
                            # No, stations are specific nodes.
                            # If we can't reach c, we skip it? 
                            # That breaks the "all customers served" constraint if we just skip.
                            # In a chunk, we assume it's feasible. 
                            # If not, we might need to insert a station.
                            # Let's just skip for now and hope LS fixes it.
                            continue
                        
                        t_arr = cur_time + d / speed
                        if t_arr < ready[c]:
                            t_arr = ready[c]
                        if t_arr > due[c]:
                            continue
                        
                        best_d = d
                        best_c = c
                
                if best_c == -1:
                    # No feasible next customer. 
                    # This is a problem. 
                    # For now, break. The remaining customers in this chunk 
                    # will be handled by a fallback insertion step later.
                    break
                
                d = dist[cur_node, best_c]
                cur_time += d / speed
                cur_bat -= d
                cur_node = best_c
                cur_load += demands[best_c]
                
                if cur_time < ready[best_c]:
                    cur_time = ready[best_c]
                cur_time += service[best_c]
                
                route.append(best_c)
                remaining.remove(best_c)
            
            if route:
                routes.append(route)
            
            # Handle leftovers from this chunk (if any)
            # They will be inserted globally in the next step.
            # We don't return them here, we just add the valid parts.
            # Wait, the global fallback in the main loop handles `unvisited`.
            # So we need to track which customers were NOT served in this construction.
        
        return routes, set(customers) - set(c for r in routes for c in r)

    def local_search_oropt_2opt(routes, time_limit):
        """
        Local search with Or-opt (move 1-3 nodes) and 2-opt.
        """
        current_total_dist = compute_total_distance(routes)
        start_ls = time.time()
        
        # Precompute route distances for delta calculation
        route_dists = [0.0] * len(routes)
        for i, r in enumerate(routes):
            _, route_dists[i] = simulate_route(r)
        
        improved = True
        no_improve_count = 0
        
        while improved and time.time() - start_ls < time_limit:
            improved = False
            no_improve_count += 1
            if no_improve_count > 50:
                break
            
            # 1. Or-opt: Move a segment of size 1, 2, or 3 from route A to route B
            for seg_size in [1, 2, 3]:
                for r1 in range(len(routes)):
                    route1 = routes[r1]
                    n1 = len(route1)
                    if n1 < seg_size:
                        continue
                    
                    for i in range(n1 - seg_size + 1):
                        segment = route1[i:i+seg_size]
                        # Remove segment
                        remaining1 = route1[:i] + route1[i+seg_size:]
                        
                        # Check validity of remaining1
                        is_valid1, new_dist1 = simulate_route(remaining1)
                        if not is_valid1:
                            continue
                        
                        # Try to insert segment into all other routes (and r1)
                        for r2 in range(len(routes)):
                            route2 = routes[r2]
                            n2 = len(route2)
                            
                            for j in range(n2 + 1):
                                # Insert segment at position j in route2
                                new_route2 = route2[:j] + segment + route2[j:]
                                is_valid2, new_dist2 = simulate_route(new_route2)
                                if not is_valid2:
                                    continue
                                
                                new_total = new_dist1 + new_dist2
                                old_total = route_dists[r1] + route_dists[r2]
                                
                                if new_total < old_total - 1e-6:
                                    routes[r1] = remaining1
                                    routes[r2] = new_route2
                                    route_dists[r1] = new_dist1
                                    route_dists[r2] = new_dist2
                                    current_total_dist = new_total
                                    improved = True
                                    break
                            
                            if improved:
                                break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                no_improve_count = 0
                continue
            
            # 2. 2-opt: Reverse segment within a route
            for r_idx in range(len(routes)):
                route = routes[r_idx]
                n = len(route)
                if n < 2:
                    continue
                
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                        is_valid, new_dist = simulate_route(new_route)
                        if is_valid and new_dist < route_dists[r_idx] - 1e-6:
                            routes[r_idx] = new_route
                            route_dists[r_idx] = new_dist
                            current_total_dist = compute_total_distance(routes) # Update total
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                no_improve_count = 0

        return routes

    # Main Execution
    best_routes = None
    best_dist = float('inf')
    
    # Multi-start construction
    # We have ~140s. 
    # Let's run 10-20 starts of Swept-Clustering with different random seeds/shuffles.
    
    num_starts = 20
    
    for i in range(num_starts):
        if time.time() > deadline - 60.0: # Keep 60s for LS
            break
        
        rng = random.Random(i)
        
        # In Swept-Clustering, the "randomness" comes from how we break ties 
        # or how we assign the remainder. 
        # To add diversity, we can rotate the starting angle.
        # Or we can just rely on the fact that different seeds might lead to 
        # different LS outcomes if we perturb the initial solution.
        
        routes, unvisited = construct_swept_cluster_solution(rng)
        
        # Handle leftovers from construction
        for c in list(unvisited):
            best_r = -1
            best_cost = float('inf')
            for r_idx, route in enumerate(routes):
                for pos in range(len(route) + 1):
                    new_route = route[:pos] + [c] + route[pos:]
                    is_valid, new_dist = simulate_route(new_route)
                    if is_valid and new_dist < best_cost:
                        best_cost = new_dist
                        best_r = r_idx
                        best_pos = pos
            
            if best_r != -1:
                routes[best_r].insert(best_pos, c)
            else:
                # New route
                routes.append([c])
            
            unvisited.remove(c)
        
        # Validate
        valid = True
        served = set()
        for route in routes:
            if not simulate_route(route)[0]:
                valid = False
                break
            for node in route:
                if node not in stations_set:
                    if node in served:
                        valid = False
                        break
                    served.add(node)
        
        if not valid or served != set(customers):
            continue
        
        d = compute_total_distance(routes)
        if d < best_dist:
            best_dist = d
            best_routes = [r[:] for r in routes]
            
        # Run local search on this solution
        if time.time() > deadline - 5.0:
            break
            
        # Allocate time for LS: proportional to remaining time
        # If we have 100s left, use 90s for LS.
        ls_time = min(30.0, deadline - time.time() - 2.0)
        if ls_time > 1.0:
            ls_routes = local_search_oropt_2opt(routes, ls_time)
            
            # Validate LS result
            valid = True
            served = set()
            for route in ls_routes:
                if not simulate_route(route)[0]:
                    valid = False
                    break
                for node in route:
                    if node not in stations_set:
                        if node in served:
                            valid = False
                            break
                        served.add(node)
            
            if valid and served == set(customers):
                d_ls = compute_total_distance(ls_routes)
                if d_ls < best_dist:
                    best_dist = d_ls
                    best_routes = [r[:] for r in ls_routes]
    
    # If no valid solution found, fallback to single greedy
    if best_routes is None:
        # Fallback: Serve each customer individually?
        # Or just return the first valid one found?
        # Let's try one more simple construction
        routes = []
        unvisited = set(customers)
        for c in sorted(customers, key=lambda x: due[x]):
            # Try to insert into existing
            inserted = False
            for r_idx in range(len(routes)):
                for pos in range(len(routes[r_idx]) + 1):
                    new_route = routes[r_idx][:pos] + [c] + routes[r_idx][:]
                    if simulate_route(new_route)[0]:
                        routes[r_idx] = new_route
                        inserted = True
                        break
                if inserted:
                    break
            if not inserted:
                routes.append([c])
            unvisited.remove(c)
        
        best_routes = routes

    return best_routes
