"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO on CVRP.
    
    Args:
        distance_matrix: (n, n) symmetric distance matrix
        coordinates: (n, 2) coordinates
        demands: (n,) demands (node 0 is depot with demand 0)
        capacity: vehicle capacity
        
    Returns:
        (n, n) heuristic matrix where larger = more attractive
    """
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s kill
    
    n = distance_matrix.shape[0]
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.inf)
    
    # Simulate ACO dynamics with cooling schedule to create a static heuristic
    # that encodes expected pheromone reinforcement trajectory
    
    # Multiple virtual "phases" representing different stages of the search
    # Early phase: high alpha (distance-focused), late phase: low alpha (demand-aware)
    phases = [
        {'alpha': 3.0, 'demand_weight': 0.0, 'weight': 0.3},  # Early: pure distance
        {'alpha': 2.0, 'demand_weight': 0.2, 'weight': 0.4},  # Mid: balanced
        {'alpha': 1.5, 'demand_weight': 0.5, 'weight': 0.3},  # Late: demand-aware
    ]
    
    demand_ratio = demands / capacity  # (n,)
    H = np.zeros((n, n))
    
    for phase in phases:
        if time.time() > deadline:
            break
        alpha = phase['alpha']
        dw = phase['demand_weight']
        w = phase['weight']
        
        # Inverse distance with phase-specific exponent
        base = 1.0 / (d ** alpha)
        
        # Demand-awareness: penalize edges to high-demand nodes
        # From depot: mild penalty (fresh capacity)
        # From customer: stronger penalty (uncertain remaining capacity)
        depot_penalty = 1.0 - 0.1 * dw * demand_ratio
        customer_penalty = 1.0 - 0.4 * dw * demand_ratio
        
        H_phase = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    H_phase[i, j] = 1e-9
                    continue
                if i == 0:
                    penalty = max(depot_penalty[j], 0.1)
                else:
                    penalty = max(customer_penalty[j], 0.1)
                H_phase[i, j] = base[i, j] * penalty
        
        H += w * H_phase
    
    # Add cluster-density term: boost edges within dense spatial regions
    if time.time() < deadline:
        # Compute k-nearest neighbor distances for each node
        k = 5
        coords = coordinates
        # For each pair (i,j), compute how many of j's k-nearest neighbors are i
        # and vice versa. This creates a local density boost.
        
        # Compute pairwise distances (already have d, but need finite values)
        d_finite = d.copy()
        np.fill_diagonal(d_finite, 0)
        
        # For each node j, find its k nearest neighbors
        density_boost = np.zeros((n, n))
        for j in range(n):
            dists_j = d_finite[:, j].copy()
            dists_j[j] = np.inf
            # Get indices of k nearest neighbors
            knn_idx = np.argsort(dists_j)[:k]
            for i in knn_idx:
                # Boost edge i->j and j->i based on proximity
                boost = 1.0 / (d_finite[i, j] + 1e-6)
                density_boost[i, j] += boost
                density_boost[j, i] += boost
        
        # Normalize density boost
        db_max = np.max(density_boost)
        if db_max > 0:
            density_boost = density_boost / db_max
        
        # Blend density boost into H with a moderate weight
        H = H + 0.15 * density_boost
    
    # Normalize to avoid numerical issues
    H_max = np.max(H)
    if H_max > 0:
        H = H / H_max
    
    # Ensure all entries are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
