import numpy as np
import time
import random

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using Simulated Annealing to optimize
    weights for a linear combination of heuristic features.
    
    Features:
    1. Inverse Distance (standard ACO heuristic)
    2. Angle Continuity (penalize sharp turns, encourage smooth paths)
    3. Demand Fit (encourage edges that keep load balanced)
    4. Cluster Affinity (soft preference for staying within k-means-like groups)
    
    The SA process minimizes the cost of a fast greedy construction heuristic,
    approximating the performance of the ACO ants.
    """
    deadline = time.time() + 320.0  # Safety margin
    n = distance_matrix.shape[0]
    
    if n == 0:
        return np.array([])
        
    # Pre-compute features
    # 1. Inverse Distance
    # Avoid division by zero for i==j
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, 1.0)
    feat_dist = 1.0 / dist_safe
    np.fill_diagonal(feat_dist, 0.0) # Diagonal will be handled later
    
    # 2. Angle Continuity
    # For each i, j, calculate the angle between vector (i->prev) and (i->j)? 
    # Actually, standard ACO heuristic is static. 
    # A better "smoothness" heuristic for a static matrix is hard.
    # Let's use a proxy: Cosine similarity of position vectors relative to depot?
    # Or simply: Prefer nodes that are closer to the "center of mass" of the remaining unvisited set?
    # Since we can't know remaining set, let's use a different feature:
    # "Radial Balance": Prefer edges that don't move too far away from the depot 
    # unless necessary, or vice versa.
    # Let's use: feat_radial = 1 / (1 + |dist_to_depot(i) - dist_to_depot(j)|)
    # This encourages visiting nodes at similar distances from depot together (layered routing).
    dist_to_depot = distance_matrix[0, :]
    diff_radial = np.abs(dist_to_depot[:, np.newaxis] - dist_to_depot[np.newaxis, :])
    feat_radial = 1.0 / (1.0 + diff_radial)
    np.fill_diagonal(feat_radial, 0.0)
    
    # 3. Demand Fit
    # Encourage edges between nodes with similar demands? 
    # Or penalize edges that would likely exceed capacity?
    # Since we don't know the order, let's use a static feature:
    # feat_demand = 1 / (1 + |demand(i) - demand(j)|)
    # This groups similar demands, which might help packing.
    demand_diff = np.abs(demands[:, np.newaxis] - demands[np.newaxis, :])
    feat_demand = 1.0 / (1.0 + demand_diff)
    np.fill_diagonal(feat_demand, 0.0)
    
    # 4. Cluster Affinity (Soft)
    # Perform a quick k-means or simple distance-based clustering to get a "cluster score".
    # We want to encourage edges within the same cluster.
    # Simple clustering: assign each node to the nearest of K random centroids?
    # Let's use a deterministic "grid" clustering or just nearest-neighbor chains.
    # For speed and robustness, let's use a simple spectral-like or distance threshold clustering.
    # Actually, let's use a simple "distance to centroid" feature.
    # Calculate centroids of K=5 clusters using k-means (simplified, fixed seeds).
    
    def quick_kmeans(coords, k, iters=10):
        # Initialize centroids as k random points
        indices = np.random.choice(len(coords), k, replace=False)
        centroids = coords[indices].copy()
        for _ in range(iters):
            # Assign
            dists = np.linalg.norm(coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(dists, axis=1)
            # Update
            for j in range(k):
                mask = labels == j
                if np.sum(mask) > 0:
                    centroids[j] = np.mean(coords[mask], axis=0)
                else:
                    # Re-initialize empty cluster
                    centroids[j] = coords[np.random.randint(len(coords))]
        # Final assignment
        dists = np.linalg.norm(coords[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        return labels

    # Use fixed seed for reproducibility within the instance
    np.random.seed(42)
    k_clusters = 5
    labels = quick_kmeans(coordinates, k_clusters)
    
    # Feature: 1 if same cluster, 0 otherwise (or distance to centroid)
    # Let's use a smooth version: 1 / (1 + |dist(i, centroid_i) - dist(j, centroid_j)|) ?
    # No, simpler: 1 if same label, else 0.1.
    feat_cluster = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                if labels[i] == labels[j]:
                    feat_cluster[i, j] = 1.0
                else:
                    feat_cluster[i, j] = 0.1
    
    # Stack features
    # F = [feat_dist, feat_radial, feat_demand, feat_cluster]
    features = np.stack([feat_dist, feat_radial, feat_demand, feat_cluster], axis=-1)
    
    # Simulated Annealing to find optimal weights w
    # Objective: Minimize cost of a greedy tour constructed using H = sum(w_k * F_k)
    
    def evaluate_weights(w):
        # Construct heuristic matrix
        # H[i, j] = sum(w_k * F[i, j, k])
        # Ensure positivity
        H = np.einsum('ijk,k->ij', features, w)
        H = np.maximum(H, 1e-9)
        np.fill_diagonal(H, 1e-9)
        
        # Fast Greedy Simulation (approximating ACO behavior)
        # This is a simplified "nearest neighbor with capacity check"
        # We run this multiple times with slight randomness to estimate robustness
        total_cost = 0.0
        runs = 5 # Few runs for speed
        
        for _ in range(runs):
            unvisited = set(range(1, n))
            current = 0
            load = 0.0
            cost = 0.0
            
            while unvisited:
                # Find next node
                best_node = None
                best_score = -1e9
                best_dist = float('inf')
                
                # To speed up, we can vectorize the selection if we maintain arrays, 
                # but for n=50, a loop is fine.
                # However, we want to mimic ACO: prob ~ pheromone * heuristic.
                # Since pheromone is not available here, we just use heuristic.
                # We pick the node with max heuristic value among feasible ones.
                
                feasible_nodes = []
                for j in unvisited:
                    if load + demands[j] <= capacity:
                        feasible_nodes.append(j)
                
                if not feasible_nodes:
                    # Return to depot
                    cost += distance_matrix[current, 0]
                    current = 0
                    load = 0.0
                    continue
                
                # Select best feasible node based on H
                # Add small random noise to break ties and estimate variance
                scores = np.array([H[current, j] for j in feasible_nodes])
                # Add noise
                noise = np.random.normal(0, 0.1, size=len(scores))
                scores += noise
                best_idx = np.argmax(scores)
                next_node = feasible_nodes[best_idx]
                
                cost += distance_matrix[current, next_node]
                load += demands[next_node]
                unvisited.remove(next_node)
                current = next_node
                
            # Return to depot
            cost += distance_matrix[current, 0]
            total_cost += cost
        
        return total_cost / runs

    # SA Parameters
    T = 100.0
    T_min = 0.01
    alpha = 0.95
    max_iter = 1000
    
    # Initial weights: uniform
    w_current = np.ones(4)
    w_current /= np.sum(w_current)
    
    cost_current = evaluate_weights(w_current)
    w_best = w_current.copy()
    cost_best = cost_current
    
    iteration = 0
    while T > T_min and iteration < max_iter:
        if time.time() > deadline - 30:
            break
            
        # Perturb weights
        w_new = w_current.copy()
        w_new += np.random.normal(0, 0.1, size=4)
        w_new = np.maximum(w_new, 0.0)
        if np.sum(w_new) > 0:
            w_new /= np.sum(w_new)
        else:
            w_new = w_current.copy()
            
        cost_new = evaluate_weights(w_new)
        
        # Acceptance criterion
        if cost_new < cost_current or np.random.random() < np.exp(-(cost_new - cost_current) / T):
            w_current = w_new
            cost_current = cost_new
            
            if cost_new < cost_best:
                cost_best = cost_new
                w_best = w_new.copy()
                
        T *= alpha
        iteration += 1
        
    # Final Heuristic Matrix using best weights
    H = np.einsum('ijk,k->ij', features, w_best)
    H = np.maximum(H, 1e-9)
    np.fill_diagonal(H, 1e-9)
    
    # Ensure shape and finiteness
    if H.shape != (n, n):
        H = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(H, 1e-9)
        
    if not np.all(np.isfinite(H)):
        H = 1.0 / (distance_matrix + 1e-9)
        np.fill_diagonal(H, 1e-9)
        
    return H
