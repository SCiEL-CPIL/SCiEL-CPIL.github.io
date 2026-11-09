"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search on TSP.
    
    Uses a combination of:
    1. Usage-based penalty (GLS core)
    2. Length-ratio based penalty (targets over-long edges)
    3. K-NN structural bias (targets geometrically implausible edges)
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    N = edge_distance.shape[0]
    
    # Extract tour edges
    tour = local_opt_tour
    if len(tour) < 2:
        return edge_distance.copy()
    
    # Get consecutive nodes in tour (including wrap-around)
    src = np.array(tour[:-1])
    dst = np.array(tour[1:])
    # Add closing edge
    src = np.append(src, tour[0])
    dst = np.append(dst, tour[-1])
    
    # Get true lengths of tour edges
    tour_lengths = edge_distance[src, dst]
    
    # Mean edge length for normalization
    mean_len = np.mean(tour_lengths)
    if mean_len < 1e-12:
        return edge_distance.copy()
    
    # --- 1. Usage-based penalty (GLS core) ---
    # Normalize usage counts
    max_usage = edge_n_used.max()
    if max_usage > 0:
        normalized_usage = edge_n_used / (1.0 + max_usage)
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # --- 2. Length-ratio based penalty ---
    # Penalize edges that are much longer than average
    # Use sigmoid to smoothly concentrate penalty on over-long edges
    length_ratio = edge_distance / mean_len
    # Sigmoid: centered at 1.0, steepness controls how sharply we penalize
    # For edges much longer than mean, ratio >> 1, sigmoid -> 1
    # For edges shorter than mean, ratio < 1, sigmoid -> 0
    sigma = 0.5
    length_penalty = 1.0 / (1.0 + np.exp(-sigma * (length_ratio - 1.5)))
    
    # --- 3. K-NN structural bias ---
    # For each node, find its K nearest neighbors
    K = 8
    # Compute distances from each node to all others
    # edge_distance is symmetric, so we can use it directly
    # For each row, find indices of K smallest distances (excluding self)
    knn_mask = np.zeros((N, N), dtype=bool)
    for i in range(N):
        # Exclude self
        row = edge_distance[i].copy()
        row[i] = np.inf
        # Get K smallest
        knn_indices = np.argpartition(row, K)[:K]
        knn_mask[i, knn_indices] = True
        knn_mask[knn_indices, i] = True  # Symmetric
    
    # An edge is "structurally poor" if it's NOT in either endpoint's KNN set
    # For tour edges, check if they are in the KNN set
    tour_edge_structural = np.zeros(N, dtype=bool)
    for idx in range(len(src)):
        u = src[idx]
        v = dst[idx]
        if not (knn_mask[u, v] or knn_mask[v, u]):
            tour_edge_structural[idx] = True
    
    # Create a matrix indicating structurally poor edges
    structural_poor = np.zeros((N, N), dtype=float)
    for idx in range(len(src)):
        if tour_edge_structural[idx]:
            u = src[idx]
            v = dst[idx]
            structural_poor[u, v] = 1.0
            structural_poor[v, u] = 1.0
    
    # Combine penalties
    # Usage penalty: proportional to normalized usage
    usage_penalty_weight = 0.15
    # Length penalty: proportional to length_penalty
    length_penalty_weight = 0.10
    # Structural penalty: binary, for structurally poor tour edges
    structural_penalty_weight = 0.20
    
    # Build the multiplier
    multiplier = np.ones((N, N), dtype=float)
    
    # Add usage penalty
    multiplier += usage_penalty_weight * normalized_usage
    
    # Add length penalty (only for edges that are part of the tour)
    # We want to penalize tour edges based on their length ratio
    for idx in range(len(src)):
        u = src[idx]
        v = dst[idx]
        lr = edge_distance[u, v] / mean_len
        lp = 1.0 / (1.0 + np.exp(-sigma * (lr - 1.5)))
        multiplier[u, v] += length_penalty_weight * lp
        multiplier[v, u] += length_penalty_weight * lp
    
    # Add structural penalty for structurally poor tour edges
    multiplier += structural_penalty_weight * structural_poor
    
    # Ensure symmetry
    multiplier = (multiplier + multiplier.T) / 2.0
    
    # Apply multiplier to distances
    result = edge_distance * multiplier
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, edge_distance)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
