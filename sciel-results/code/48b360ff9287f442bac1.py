"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.cluster.vq import kmeans2

def _proxy_aco(distance_matrix, demands, capacity, heuristic, n_ants=20, n_iters=30, seed=42, deadline=None):
    """Run a proxy ACO to evaluate a heuristic's quality."""
    rng = np.random.default_rng(seed)
    n = distance_matrix.shape[0]
    dm = distance_matrix
    eta = heuristic
    tau = np.ones((n, n))
    best_cost = np.inf
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        iter_best = np.inf
        for _ in range(n_ants):
            current = 0
            load = 0.0
            visited = set()
            route_cost = 0.0
            while True:
                # Find feasible unvisited customers
                feasible = []
                for j in range(1, n):
                    if j not in visited and demands_arr[j] <= capacity - load:
                        feasible.append(j)
                if not feasible:
                    # Return to depot
                    route_cost += dm[current, 0]
                    break
                # Compute probabilities
                probs = np.zeros(len(feasible))
                for idx, j in enumerate(feasible):
                    probs[idx] = tau[current, j] * eta[current, j]
                total = probs.sum()
                if total <= 0 or not np.isfinite(total):
                    probs = np.ones(len(feasible))
                    total = probs.sum()
                probs /= total
                # Select next node
                r = rng.random()
                cumsum = np.cumsum(probs)
                idx = np.searchsorted(cumsum, r)
                if idx >= len(feasible):
                    idx = len(feasible) - 1
                next_node = feasible[idx]
                route_cost += dm[current, next_node]
                load += demands_arr[next_node]
                visited.add(next_node)
                current = next_node
            if route_cost < iter_best:
                iter_best = route_cost
        if iter_best < best_cost:
            best_cost = iter_best
        # Update pheromone
        tau *= 0.9
        tau += 1.0 / max(iter_best, 1e-9)
    
    return best_cost

def _compute_cluster_heuristic(distance_matrix, coordinates, demands, capacity, n_clusters, gamma, cross_penalty=0.1):
    """
    Compute a heuristic matrix based on K-Means clustering.
    Edges within the same cluster get full weight 1/d^gamma.
    Edges between clusters get weight cross_penalty/d^gamma.
    """
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    
    # Base distance heuristic
    with np.errstate(divide='ignore', invalid='ignore'):
        base_eta = 1.0 / np.power(dm + 1e-9, gamma)
    
    # Ensure positive and finite
    base_eta[~np.isfinite(base_eta)] = 1e-9
    base_eta[base_eta <= 0] = 1e-9
    
    # If n_clusters is 1 or invalid, return base heuristic
    if n_clusters <= 1 or n_clusters >= n:
        np.fill_diagonal(base_eta, 1e-9)
        return base_eta

    # Perform K-Means clustering on customers (nodes 1 to n-1)
    # Depot (0) is not clustered, or we can assign it to the nearest cluster center?
    # For the heuristic, we need cluster labels for all nodes.
    # Let's cluster customers 1..n-1.
    customer_coords = coords[1:]
    
    try:
        # kmeans2 requires at least 1 sample. 
        # If n_clusters > number of customers, reduce it.
        actual_k = min(n_clusters, n - 1)
        if actual_k <= 1:
            np.fill_diagonal(base_eta, 1e-9)
            return base_eta
            
        # Use a fixed seed for reproducibility in clustering
        centroids, labels = kmeans2(customer_coords, actual_k, minit='points', seed=42)
        
        # Assign depot (0) to the cluster of the nearest centroid? 
        # Or just leave it with a unique label that matches no one?
        # If depot matches no one, edges from depot are penalized. 
        # Better: assign depot to the cluster of the closest customer? 
        # Or just treat depot as part of all clusters? 
        # Let's assign depot to the cluster of the nearest customer.
        dists_to_centroids = np.linalg.norm(centroids - coords[0], axis=1)
        depot_cluster = np.argmin(dists_to_centroids)
        
        # Create full label array
        full_labels = np.zeros(n, dtype=int)
        full_labels[0] = depot_cluster
        full_labels[1:] = labels
        
        # Apply penalty
        heuristic = base_eta.copy()
        for i in range(n):
            for j in range(n):
                if i != j and full_labels[i] != full_labels[j]:
                    heuristic[i, j] *= cross_penalty
        
        np.fill_diagonal(heuristic, 1e-9)
        return heuristic
        
    except Exception:
        # Fallback to base heuristic if clustering fails
        np.fill_diagonal(base_eta, 1e-9)
        return base_eta

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix for ACO using a Cluster-Based structural approach.
    Searches for optimal number of clusters and distance exponent.
    """
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    cap = float(capacity)
    
    # Estimate minimum number of vehicles
    total_demand = np.sum(dem[1:])
    min_vehicles = max(1, int(np.ceil(total_demand / cap)))
    
    # Define search space for structural parameters
    # K: number of clusters
    # Gamma: distance exponent
    # We test a few K values around the theoretical minimum
    k_values = [min_vehicles, min_vehicles + 1, min_vehicles + 2]
    gamma_values = [1.0, 1.5, 2.0, 2.5]
    
    best_score = np.inf
    best_heuristic = None
    best_params = (min_vehicles, 2.0)
    
    seed = 42
    
    # Iterate through structural configurations
    for k in k_values:
        for gamma in gamma_values:
            if time.time() > deadline - 10.0:
                break
            
            # Compute heuristic for this configuration
            # We use a cross_penalty of 0.1 to strongly encourage intra-cluster travel
            eta = _compute_cluster_heuristic(dm, coords, dem, cap, k, gamma, cross_penalty=0.1)
            
            # Validate heuristic
            if not np.all(np.isfinite(eta)) or eta.shape != dm.shape:
                continue
                
            # Run proxy ACO to evaluate
            score = _proxy_aco(dm, dem, cap, eta, 
                               n_ants=20, n_iters=30, seed=seed, 
                               deadline=deadline - 5.0)
            
            if score < best_score:
                best_score = score
                best_heuristic = eta.copy()
                best_params = (k, gamma)
                
    # If no valid heuristic was found (should not happen), fall back to 1/d
    if best_heuristic is None:
        with np.errstate(divide='ignore', invalid='ignore'):
            best_heuristic = 1.0 / np.power(dm + 1e-9, 2.0)
        best_heuristic[~np.isfinite(best_heuristic)] = 1e-9
        best_heuristic[best_heuristic <= 0] = 1e-9
        np.fill_diagonal(best_heuristic, 1e-9)
        
    return best_heuristic
# EVOLVE-BLOCK-END
