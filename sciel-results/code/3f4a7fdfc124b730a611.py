import math
import time
import random
import numpy as np
from scipy.spatial.distance import cdist

_EPS = 1e-9

def plan(instance):
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin within 150s kill

    coords = np.array(instance["coords"], dtype=float)
    demands = np.array(instance["demands"], dtype=float)
    ready = np.array(instance["ready"], dtype=float)
    due = np.array(instance["due"], dtype=float)
    service = np.array(instance["service"], dtype=float)
    n_cust = instance["n_customers"]
    stations = list(instance["stations"])
    capacity = instance["capacity"]
    battery_cap = instance["battery"]
    speed = instance["speed"]
    recharge_rate = instance["recharge_rate"]
    horizon = instance["horizon"]

    # Precompute distance matrix
    n_nodes = len(coords)
    dist = cdist(coords, coords)

    customers = list(range(1, n_cust + 1))
    
    # Helper to evaluate a route: returns (valid, distance, time)
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0
        t = 0.0
        b = battery_cap
        load = 0.0
        d_total = 0.0
        prev = 0
        for node in route_nodes:
            d = dist[prev, node]
            t += d / speed
            b -= d
            if b < -_EPS:
                return False, 0.0, 0.0
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, 0.0
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, 0.0
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, 0.0
        if t > horizon + _EPS:
            return False, 0.0, 0.0
        d_total += d
        return True, d_total, t

    def evaluate_routes(routes):
        """Returns (valid, total_distance) for a list of routes."""
        total_d = 0.0
        for r in routes:
            if not r: continue
            valid, d, _ = check_route(r)
            if not valid:
                return False, 0.0
            total_d += d
        return True, total_d

    # --- Strategy: Cluster-First, Route-Second ---
    
    # 1. Determine number of clusters
    # Capacity 100, Avg demand ~15 -> ~6-7 customers per route max by capacity.
    # Battery 300, Max dist ~141. 3-4 customers per route is safer for battery.
    # Let's aim for 5-7 routes.
    # We will try a few different cluster counts and keep the best valid result.
    
    best_overall_routes = None
    best_overall_dist = float('inf')
    
    # Initial simple valid solution: one customer per route
    fallback_routes = [[c] for c in customers]
    valid, dist = evaluate_routes(fallback_routes)
    if valid:
        best_overall_routes = [list(r) for r in fallback_routes]
        best_overall_dist = dist

    # Try different numbers of clusters
    # Since we have 25 customers, trying 4 to 8 clusters covers a good range.
    for k in range(4, 9):
        if time.time() > DEADLINE - 20: break
        
        # 2. K-Means Clustering
        # Use scipy or simple numpy implementation
        # Initialize centroids randomly or using k-medoids
        rng = np.random.default_rng(42 + k)
        
        # Simple K-Means
        # Pick k random customers as initial centroids
        indices = rng.choice(n_cust, k, replace=False)
        centroids = coords[indices + 1] # +1 because customer 1 is index 1 in coords
        
        labels = np.zeros(n_cust, dtype=int)
        
        for _ in range(20): # Max iterations
            # Assign points to nearest centroid
            d_mat = cdist(coords[1:n_cust+1], centroids)
            new_labels = np.argmin(d_mat, axis=1)
            
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Update centroids
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    centroids[i] = np.mean(coords[1:n_cust+1][mask], axis=0)
                else:
                    # Reinitialize empty cluster to a random customer
                    idx = rng.integers(0, n_cust)
                    centroids[i] = coords[idx + 1]
        
        # 3. Formulate Initial Routes from Clusters
        # Each cluster becomes a route.
        # Order customers in each route using Nearest Neighbor from Depot.
        
        routes = []
        for i in range(k):
            cluster_customers = [j + 1 for j in range(n_cust) if labels[j] == i]
            if not cluster_customers:
                continue
            
            # Check if cluster demand exceeds capacity?
            # If so, we might need to split, but for now assume K is large enough.
            # Actually, if demand > capacity, the route is invalid.
            # We should probably check this and split if needed, or just mark invalid.
            # Given small demands (5-25) and capacity 100, and k>=4, it's unlikely a single cluster exceeds 100 unless k is small and demands are high.
            # If invalid, we will just skip this k or handle later.
            
            # Nearest Neighbor construction
            route = []
            unvisited = list(cluster_customers)
            curr = 0
            while unvisited:
                # Find nearest unvisited
                d_to_unvisited = dist[curr, unvisited]
                min_idx = np.argmin(d_to_unvisited)
                next_node = unvisited[min_idx]
                route.append(next_node)
                unvisited.pop(min_idx)
                curr = next_node
            
            # Check validity
            valid, d, _ = check_route(route)
            if valid:
                routes.append(route)
            else:
                # If invalid, fallback to one customer per route for this cluster
                # Or try to split. For simplicity, if a cluster is invalid, 
                # we add its customers as separate routes.
                for c in cluster_customers:
                    routes.append([c])
        
        # 4. Local Search: 2-Opt and Or-Opt within each route
        # And Relocate between routes
        
        current_routes = [list(r) for r in routes]
        valid, current_dist = evaluate_routes(current_routes)
        if not valid:
            continue
            
        improved = True
        max_ls_iters = 500
        ls_iter = 0
        
        while improved and ls_iter < max_ls_iters and time.time() < DEADLINE - 10:
            improved = False
            ls_iter += 1
            
            # 2-Opt within each route
            for i in range(len(current_routes)):
                r = current_routes[i]
                if len(r) < 2:
                    continue
                
                # Standard 2-Opt
                for j in range(len(r) - 1):
                    for k in range(j + 1, len(r)):
                        # Calculate delta for reversing r[j+1..k]
                        # d(a,b) + d(c,d) vs d(a,c) + d(b,d)
                        # a = r[j], b = r[j+1], c = r[k], d = r[k+1] (or depot)
                        
                        # Actually, simpler to just try and check validity/cost
                        # But that's O(N^3).
                        # Let's use the delta calculation for speed.
                        
                        a = r[j]
                        b = r[j+1]
                        c = r[k]
                        d = r[k+1] if k+1 < len(r) else 0
                        
                        old_cost = dist[a, b] + dist[c, d]
                        new_cost = dist[a, c] + dist[b, d]
                        
                        if new_cost < old_cost - _EPS:
                            # Check if reverse is valid (time windows/battery)
                            # We need to simulate the route to be sure.
                            # Optimization: Only check if it looks promising.
                            # For now, let's do a quick check by constructing the new route.
                            new_r = r[:j+1] + r[j+1:k+1][::-1] + r[k+1:]
                            v, d_new, _ = check_route(new_r)
                            if v:
                                # Check total distance change
                                # We calculated delta for distance, but time/battery might fail.
                                # If it passes check_route, it's valid.
                                # We need to ensure the TOTAL distance decreased.
                                # The delta calculation above assumed Euclidean 2-opt.
                                # If valid, we accept it.
                                current_routes[i] = new_r
                                improved = True
                                break
                    if improved:
                        break
            
            if improved:
                continue
                
            # Or-Opt: Move 1 or 2 nodes to a better position
            for i in range(len(current_routes)):
                r = current_routes[i]
                if len(r) < 3:
                    continue
                
                for j in range(len(r)):
                    # Move node r[j]
                    node = r[j]
                    new_r = r[:j] + r[j+1:]
                    
                    # Try inserting at all other positions
                    best_pos = -1
                    best_d = float('inf')
                    
                    for pos in range(len(new_r) + 1):
                        if pos == j or (pos > 0 and new_r[pos-1] == node): # Skip original pos roughly
                            continue
                        trial = new_r[:pos] + [node] + new_r[pos:]
                        v, d_trial, _ = check_route(trial)
                        if v and d_trial < best_d:
                            best_d = d_trial
                            best_pos = pos
                    
                    if best_pos != -1:
                        # Check if this improves the route distance
                        # We need old route distance
                        v_old, d_old, _ = check_route(r)
                        if d_trial < d_old - _EPS:
                            current_routes[i] = new_r
                            improved = True
                            break
                if improved:
                    break
            
            if improved:
                continue

            # Relocate: Move a node from route i to route j
            moved = False
            for i in range(len(current_routes)):
                if len(current_routes[i]) < 2:
                    continue
                for j in range(len(current_routes[i])):
                    node = current_routes[i][j]
                    src_r = current_routes[i]
                    new_src = src_r[:j] + src_r[j+1:]
                    
                    for m in range(len(current_routes)):
                        if m == i:
                            continue
                        dst_r = current_routes[m]
                        # Try inserting node into dst_r at all positions
                        best_pos = -1
                        best_total_d = float('inf')
                        
                        # Calculate current total distance of src and dst
                        v_s, d_s, _ = check_route(new_src)
                        if not v_s:
                            continue
                        v_d, d_d, _ = check_route(dst_r)
                        if not v_d:
                            continue
                        
                        base_d = d_s + d_d
                        
                        for pos in range(len(dst_r) + 1):
                            trial_dst = dst_r[:pos] + [node] + dst_r[pos:]
                            v, d_trial_dst, _ = check_route(trial_dst)
                            if v:
                                total_d = d_s + d_trial_dst
                                if total_d < best_total_d:
                                    best_total_d = total_d
                                    best_pos = pos
                        
                        if best_pos != -1 and best_total_d < base_d - _EPS:
                            # Perform move
                            current_routes[i] = new_src
                            current_routes[m] = current_routes[m][:best_pos] + [node] + current_routes[m][best_pos:]
                            moved = True
                            break
                    if moved:
                        break
                if moved:
                    break
            
            if moved:
                # Remove empty routes
                current_routes = [r for r in current_routes if r]
                continue

        # After local search, evaluate
        valid, final_dist = evaluate_routes(current_routes)
        if valid and final_dist < best_overall_dist:
            best_overall_dist = final_dist
            best_overall_routes = [list(r) for r in current_routes]

    # Final check
    if best_overall_routes is None:
        return fallback_routes
        
    return best_overall_routes
