import numpy as np
import scipy.cluster.vq as vq
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = len(demands)
    start_time = time.time()
    
    # 1. Estimate number of vehicles (clusters)
    total_demand = np.sum(demands)
    K = max(2, int(np.ceil(total_demand / capacity)))
    # Cap K to avoid too many tiny clusters if n is small, but n=50 is fine.
    # Also, K cannot exceed n-1 (customers)
    K = min(K, n - 1)
    
    # 2. Cluster customers (indices 1 to n-1)
    # coordinates[0] is depot, so we cluster coordinates[1:]
    customer_coords = coordinates[1:]
    
    # Use kmeans from scipy
    # kmeans(data, k, iter=10, minit='points')
    # data shape (N-1, 2)
    try:
        centroids, dists, _ = vq.kmeans(customer_coords, K, iter=10, minit='points')
        # Assign each customer to the nearest centroid
        labels, _ = vq.vq(customer_coords, centroids)
        # labels[0] corresponds to customer 1, labels[1] to customer 2, etc.
        # We create a cluster_id array of size n, with depot (0) having a special ID (e.g., -1)
        cluster_ids = np.full(n, -1, dtype=int)
        cluster_ids[1:] = labels
    except Exception:
        # Fallback if kmeans fails: all in one cluster
        cluster_ids = np.zeros(n, dtype=int)
        cluster_ids[0] = -1

    # 3. Define heuristic function
    # We will try a few betas and pick the best one using a proxy solver
    # that respects the cluster structure? 
    # Actually, the proxy solver in the previous code was a simple greedy.
    # A greedy solver that respects clusters (only move to same cluster) would be a good proxy.
    
    betas = [1.0, 1.5, 2.0]
    
    def calculate_heuristic(beta):
        # d_safe to avoid div by zero
        d_safe = np.where(distance_matrix > 1e-9, distance_matrix, 1.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            eta = 1.0 / (d_safe ** beta)
        
        # Apply Cluster Mask
        # For i, j:
        # If i == 0 or j == 0: keep eta
        # Else if cluster_ids[i] == cluster_ids[j]: keep eta
        # Else: set to 1e-9
        
        # Vectorized approach:
        # Create a mask of shape (n, n)
        # mask[i, j] is True if (i==0) or (j==0) or (cluster_ids[i] == cluster_ids[j])
        
        # To avoid large memory usage (50x50 is small, so it's fine), we can do it simply.
        # But let's be efficient.
        
        # Initialize eta as calculated
        # Create a boolean mask for "allowed" moves
        # allowed[i, j] = True if connection is permitted
        
        # We can construct the mask using broadcasting
        # ci = cluster_ids[:, None]
        # cj = cluster_ids[None, :]
        # is_depot_i = (np.arange(n) == 0)[:, None]
        # is_depot_j = (np.arange(n) == 0)[None, :]
        # allowed = is_depot_i | is_depot_j | (ci == cj)
        
        # However, we must ensure that if a move is NOT allowed, eta becomes 1e-9.
        # The problem: if we set eta to 1e-9, the ant might still pick it if no other option?
        # The ACO rule: "picks an unvisited customer whose demand fits... probability proportional to...".
        # If all remaining candidates in the same cluster are full/visited, but there are others in other clusters,
        # the ant should return to depot.
        # If we set cross-cluster eta to 1e-9, the probability is negligible.
        # BUT, if the ant is in cluster A, and all customers in A are visited, but customers in B are not,
        # and the ant hasn't returned to depot, it will try to go to B.
        # The heuristic will be 1e-9. Pheromone might be high?
        # If pheromone is high, it might still go to B.
        # To strictly enforce the structure, we might want to make it impossible?
        # But the task says "returns an edge-attractiveness matrix". It doesn't say we can block edges.
        # However, 1e-9 is effectively blocking it unless pheromone is huge.
        # Given pheromone starts at 1 and decays, and is reinforced by 1/cost,
        # 1e-9 is very small.
        
        # Let's apply the mask
        idx = np.arange(n)
        is_depot_i = (idx == 0)
        is_depot_j = (idx == 0)
        
        # Broadcast to (n, n)
        # allowed[i, j] is True if i is depot OR j is depot OR cluster[i] == cluster[j]
        # Note: cluster_ids[0] is -1.
        # If i=0, allowed. If j=0, allowed.
        # If i>0, j>0, check cluster match.
        
        ci = cluster_ids[:, None]
        cj = cluster_ids[None, :]
        
        # Logic:
        # allowed = (i==0) | (j==0) | (ci==cj)
        # We can compute this matrix
        
        allowed_mask = np.zeros((n, n), dtype=bool)
        # Set rows and cols for depot to True
        allowed_mask[0, :] = True
        allowed_mask[:, 0] = True
        
        # For customer-customer pairs
        # We need to check ci == cj for i>0, j>0
        # We can just compute the full matrix and overwrite depot rows/cols
        
        match_mask = (ci == cj)
        # The depot has -1, so depot-depot is True, depot-customer is False (unless customer is -1, which it isn't).
        # So match_mask[0, 0] is True.
        # match_mask[0, j>0] is False.
        # But we already set allowed_mask[0, :] and allowed_mask[:, 0] to True.
        
        # Combine
        allowed_mask = allowed_mask | match_mask
        
        # Apply to eta
        eta[~allowed_mask] = 1e-9
        
        # Handle non-finite and non-positive
        eta[~np.isfinite(eta)] = 1e-9
        eta[eta <= 0] = 1e-9
        
        return eta

    def proxy_solve(eta):
        # Greedy construction using eta
        # This proxy should ideally respect the cluster structure implicitly by following high eta
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        total_dist = 0.0
        current = 0
        load = 0.0
        
        for _ in range(n - 1):
            best_j = -1
            best_score = -1.0
            
            # Optimization: Pre-compute valid candidates? 
            # N=50 is small.
            for j in range(1, n):
                if not visited[j]:
                    if load + demands[j] <= capacity:
                        score = eta[current, j]
                        if score > best_score:
                            best_score = score
                            best_j = j
            
            if best_j == -1:
                # Return to depot
                total_dist += distance_matrix[current, 0]
                current = 0
                load = 0.0
            else:
                total_dist += distance_matrix[current, best_j]
                load += demands[best_j]
                visited[best_j] = True
                current = best_j
        
        if current != 0:
            total_dist += distance_matrix[current, 0]
            
        return total_dist

    best_beta = 1.0
    best_cost = float('inf')
    
    for beta in betas:
        if time.time() - start_time > 280: # Safety margin
            break
        eta = calculate_heuristic(beta)
        cost = proxy_solve(eta)
        if cost < best_cost:
            best_cost = cost
            best_beta = beta
            
    return calculate_heuristic(best_beta)
