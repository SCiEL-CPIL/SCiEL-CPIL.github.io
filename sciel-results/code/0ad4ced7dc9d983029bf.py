"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness heuristic matrix for ACO TSP.
    
    Strategy: Spectral graph-theoretic heuristic using MST structure.
    - Compute MST using Prim's algorithm
    - Use node degrees in MST as a measure of "hub-ness"
    - Combine with exponential decay based on distance rank
    - This creates a heuristic that favors edges connecting high-degree MST nodes
      while still respecting local distance structure
    """
    start_time = time.time()
    deadline = 290.0  # safety margin
    
    n = distance_matrix.shape[0]
    if n < 3:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Check deadline early
    if time.time() - start_time > deadline - 10.0:
        return 1.0 / (distance_matrix + 1e-9)
    
    # Step 1: Compute MST using Prim's algorithm
    mst_adj = np.zeros((n, n), dtype=np.float64)
    in_mst = np.zeros(n, dtype=bool)
    min_dist = distance_matrix[0].copy()
    in_mst[0] = True
    parent = np.full(n, -1, dtype=np.int64)
    
    for _ in range(n - 1):
        # Find the closest node not yet in MST
        # Set distances of already-in-MST nodes to inf
        temp_dist = min_dist.copy()
        temp_dist[in_mst] = np.inf
        u = np.argmin(temp_dist)
        
        if min_dist[u] == np.inf:
            break  # Disconnected graph (shouldn't happen)
        
        in_mst[u] = True
        mst_adj[u, parent[u]] = min_dist[u]
        mst_adj[parent[u], u] = min_dist[u]
        
        # Update min_dist
        for v in range(n):
            if not in_mst[v] and distance_matrix[u, v] < min_dist[v]:
                min_dist[v] = distance_matrix[u, v]
                parent[v] = u
    
    # Step 2: Compute node degrees in MST
    mst_degree = np.sum(mst_adj > 0, axis=1)
    # Add small offset to avoid zero
    mst_degree = mst_degree + 1.0
    
    # Step 3: Compute distance ranks for each edge
    # For each node, rank its neighbors by distance
    dist_ranks = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        sorted_indices = np.argsort(dists)
        for rank, j in enumerate(sorted_indices):
            dist_ranks[i, j] = rank
    
    # Step 4: Compute the spectral heuristic
    # Edge attractiveness = (degree_i * degree_j) * exp(-beta * (rank_ij + rank_ji) / 2)
    # This favors edges between high-degree MST nodes that are locally close
    
    beta = 0.15  # decay rate for distance rank
    
    # Average rank for edge (i,j)
    avg_rank = (dist_ranks + dist_ranks.T) / 2.0
    
    # Product of degrees
    degree_product = np.outer(mst_degree, mst_degree)
    
    # Exponential decay based on average rank
    exp_decay = np.exp(-beta * avg_rank)
    
    # Combine
    heuristic = degree_product * exp_decay
    
    # Also incorporate the actual distance as a baseline
    # Use a soft combination: heuristic = base_heur * (1 + spectral_component)
    base_heur = 1.0 / (distance_matrix + 1e-9)
    
    # Normalize spectral component to have similar scale as base_heur
    spectral_max = np.max(heuristic)
    if spectral_max > 0:
        heuristic_normalized = heuristic / spectral_max
    else:
        heuristic_normalized = heuristic
    
    # Combine: use a weighted sum
    # The base heuristic captures local distance, spectral captures global structure
    combined = base_heur * (1.0 + 5.0 * heuristic_normalized)
    
    # Ensure diagonal is 0
    np.fill_diagonal(combined, 0.0)
    
    # Ensure all values are positive and finite
    combined = np.where(np.isfinite(combined) & (combined > 0), combined, 1e-9)
    
    # Check deadline
    if time.time() - start_time > deadline - 5.0:
        return 1.0 / (distance_matrix + 1e-9)
    
    return combined
# EVOLVE-BLOCK-END
