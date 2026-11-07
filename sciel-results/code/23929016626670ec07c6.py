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
    
    # Helper to check if a route is valid and return its cost
    def check_route(route_nodes):
        if not route_nodes:
            return True, 0.0, 0.0, 0.0, 0.0
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
                return False, 0.0, t, d_total, load
            t = max(t, ready[node])
            if t > due[node] + _EPS:
                return False, 0.0, t, d_total, load
            t += service[node]
            load += demands[node]
            if load > capacity + _EPS:
                return False, 0.0, t, d_total, load
            d_total += d
            prev = node
        # Return to depot
        d = dist[prev, 0]
        t += d / speed
        b -= d
        if b < -_EPS:
            return False, 0.0, t, d_total, load
        if t > horizon + _EPS:
            return False, 0.0, t, d_total, load
        d_total += d
        return True, d_total, t, d_total, load

    # Step 1: Cluster-First
    # Determine number of clusters (routes) based on capacity and demand
    total_demand = sum(demands[c] for c in customers)
    min_routes = max(1, int(np.ceil(total_demand / capacity)))
    
    # Try a few numbers of clusters around min_routes
    # K-Means implementation
    def kmeans_cluster(points, k, n_iter=10, seed=42):
        rng = np.random.RandomState(seed)
        # Initialize centroids randomly from points
        centroids = points[rng.choice(len(points), k, replace=False)]
        
        for _ in range(n_iter):
            # Assign points to nearest centroid
            distances = cdist(points, centroids)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = labels == i
                if np.sum(mask) > 0:
                    new_centroids[i] = np.mean(points[mask], axis=0)
                else:
                    # If empty, keep old centroid or pick random point
                    new_centroids[i] = centroids[i]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
            
        # Final assignment
        distances = cdist(points, centroids)
        labels = np.argmin(distances, axis=1)
        
        clusters = [[] for _ in range(k)]
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
            
        return clusters

    # Try different K values
    best_routes = None
    best_score = float('inf')
    
    # We need to map cluster indices back to customer IDs
    cust_coords = coords[1:n_cust+1]
    
    # Try K from min_routes to min_routes + 2
    for k in range(min_routes, min_routes + 3):
        if time.time() > DEADLINE:
            break
            
        clusters_idx = kmeans_cluster(cust_coords, k, n_iter=20, seed=42)
        
        # Convert index clusters to customer ID clusters
        clusters = []
        for c_idx in clusters_idx:
            if not c_idx:
                continue
            cust_ids = [1 + i for i in c_idx]
            # Check capacity
            total_dem = sum(demands[c] for c in cust_ids)
            if total_dem > capacity:
                # If over capacity, we might need more clusters or split.
                # For simplicity in this heuristic, if over capacity, we mark this K as invalid
                # and rely on the merge phase or try higher K.
                # Actually, kmeans doesn't respect capacity. 
                # Let's just proceed and let check_route fail, or we can do a simple split.
                # Given 25 customers and cap 100, min_routes is likely 3-5.
                # If a cluster is too big, the SA will struggle.
                # Let's just filter out clusters that are definitely invalid? No, SA might fix order but not demand.
                # Demand is fixed per customer. So if sum(demands) > capacity, it's invalid regardless of order.
                # So we must ensure clusters respect capacity.
                # Simple fix: if capacity violated, split the cluster arbitrarily? 
                # Or just try a higher K.
                # Let's just skip this K if any cluster violates capacity.
                break
            clusters.append(cust_ids)
        
        if not clusters:
            continue
            
        # If some customers are missing (due to empty clusters or capacity break), add them as singletons
        served = set()
        for c in clusters:
            for cust in c:
                served.add(cust)
        missing = set(customers) - served
        for m in missing:
            clusters.append([m])
            
        # Step 2: Route-Second (Simulated Annealing per cluster)
        # For each cluster, find the best sequence using SA
        optimized_routes = []
        
        for cluster in clusters:
            if not cluster:
                continue
                
            # Initial sequence: sort by due date
            seq = sorted(cluster, key=lambda c: due[c])
            
            # SA Parameters
            T = 10.0
            T_min = 0.01
            alpha = 0.995
            max_iter = 2000
            
            current_seq = list(seq)
            current_valid, current_dist, _, _, _ = check_route(current_seq)
            if not current_valid:
                # If initial is invalid, try random shuffles until valid or give up
                for _ in range(100):
                    random.shuffle(current_seq)
                    current_valid, current_dist, _, _, _ = check_route(current_seq)
                    if current_valid:
                        break
                if not current_valid:
                    # Fallback: single customer routes for this cluster?
                    # Or just keep the best effort?
                    # If we can't find a valid route, we must split.
                    # For now, let's just append the sequence as is, hoping the merge phase or final check handles it?
                    # No, invalid route makes whole solution invalid.
                    # We must ensure validity.
                    # If SA fails to find valid, we split the cluster into singletons.
                    for c in current_seq:
                        optimized_routes.append([c])
                    continue
                    
            best_seq = list(current_seq)
            best_dist = current_dist
            
            for i in range(max_iter):
                if time.time() > DEADLINE:
                    break
                if T < T_min:
                    break
                    
                # Move: Swap two random elements
                new_seq = list(current_seq)
                if len(new_seq) > 1:
                    i1, i2 = random.sample(range(len(new_seq)), 2)
                    new_seq[i1], new_seq[i2] = new_seq[i2], new_seq[i1]
                elif len(new_seq) == 1:
                    # No move possible
                    T *= alpha
                    continue
                else:
                    T *= alpha
                    continue
                    
                new_valid, new_dist, _, _, _ = check_route(new_seq)
                
                if new_valid:
                    delta = new_dist - current_dist
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_seq = new_seq
                        current_dist = new_dist
                        if current_dist < best_dist:
                            best_dist = current_dist
                            best_seq = list(current_seq)
                else:
                    # If invalid, we penalize heavily
                    # For simplicity, we only accept valid moves in this implementation
                    # A more robust SA would use a penalty for invalidity
                    pass
                    
                T *= alpha
                
            optimized_routes.append(best_seq)
            
        # Step 3: Greedy Merge
        # Try to merge routes to reduce total distance
        # Sort routes by length? Or just try all pairs?
        # With ~5-10 routes, O(N^2) is fine.
        
        # Calculate initial total distance
        total_dist = 0.0
        for r in optimized_routes:
            valid, d, _, _, _ = check_route(r)
            if valid:
                total_dist += d
            else:
                # Should not happen if SA worked, but just in case
                total_dist += float('inf')
                
        if best_routes is None or total_dist < best_score:
            best_score = total_dist
            best_routes = [list(r) for r in optimized_routes]
            
        # Attempt Merges
        merged = True
        while merged:
            merged = False
            for i in range(len(best_routes)):
                for j in range(len(best_routes) - 1, i, -1):
                    if i == j:
                        continue
                    r1 = best_routes[i]
                    r2 = best_routes[j]
                    
                    # Try inserting each customer of r2 into r1
                    # This is O(N1 * N2 * N1) which is small.
                    # We want to see if we can merge r2 into r1 and keep r1 valid.
                    # If so, remove r2.
                    
                    # Try all permutations? No, too expensive.
                    # Try inserting customers of r2 one by one into best position in r1.
                    # If all fit, merge.
                    
                    trial_r1 = list(r1)
                    success = True
                    for c in r2:
                        best_pos = -1
                        best_d = float('inf')
                        for pos in range(len(trial_r1) + 1):
                            trial = trial_r1[:pos] + [c] + trial_r1[pos:]
                            valid, d, _, _, _ = check_route(trial)
                            if valid and d < best_d:
                                best_d = d
                                best_pos = pos
                        if best_pos != -1:
                            trial_r1 = trial_r1[:best_pos] + [c] + trial_r1[best_pos:]
                        else:
                            success = False
                            break
                    
                    if success:
                        # Check if total distance improved
                        d_r1_old, _, _, _, _ = check_route(r1)
                        d_r2_old, _, _, _, _ = check_route(r2)
                        d_r1_new, _, _, _, _ = check_route(trial_r1)
                        
                        if d_r1_new < d_r1_old + d_r2_old - _EPS:
                            best_routes[i] = trial_r1
                            best_routes.pop(j)
                            merged = True
                            break
                if merged:
                    break
                    
        # Update best score after merges
        final_total_dist = 0.0
        for r in best_routes:
            valid, d, _, _, _ = check_route(r)
            if valid:
                final_total_dist += d
            else:
                final_total_dist += float('inf')
                
        if final_total_dist < best_score:
            best_score = final_total_dist
            # best_routes is already updated
            
    # Final Validation and Return
    if best_routes is None:
        # Fallback: single customer routes
        return [[c] for c in customers]
        
    final_routes = []
    for r in best_routes:
        if r:
            final_routes.append(r)
            
    # Ensure all customers are served
    served = set()
    for r in final_routes:
        for c in r:
            served.add(c)
            
    missing = set(customers) - served
    if missing:
        for c in missing:
            final_routes.append([c])
            
    return final_routes
