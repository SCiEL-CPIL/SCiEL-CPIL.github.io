# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using a repulsive force model.
    
    This implementation applies an additive penalty to edges that are inversely 
    proportional to their Euclidean distance. This ensures that short edges, 
    which are typically preferred by greedy heuristics like 2-opt, receive 
    disproportionately large penalties if they have been part of previous 
    local optima. This forces the search to explore alternative, potentially 
    longer, connections to escape local optima.
    
    The penalty consists of two parts:
    1. A cumulative penalty based on edge_n_used, which accumulates over rounds.
    2. An immediate penalty for edges in the current local_opt_tour, to ensure 
       the next search step diverges from the current solution.
    """
    n = edge_distance.shape[0]
    
    # Ensure we have valid inputs
    if edge_distance is None or edge_distance.shape[0] != edge_distance.shape[1]:
        return edge_distance.copy()
    
    # Create a copy to avoid modifying the input
    penalized = edge_distance.copy()
    
    # Extract the diagonal (distance 0) and set it to 0 in the penalty matrix
    np.fill_diagonal(penalized, 0.0)
    
    # Small epsilon to avoid division by zero for very short edges
    eps = 1e-6
    
    # Calculate the inverse distance weight
    # This creates the "repulsive force": short edges get large weights
    inv_dist = 1.0 / (edge_distance + eps)
    np.fill_diagonal(inv_dist, 0.0)
    
    # 1. Cumulative penalty from edge_n_used
    # We scale the usage count by the inverse distance
    # lambda_cum controls the strength of the historical penalty
    lambda_cum = 0.01
    usage = edge_n_used.astype(float)
    # Avoid division issues if usage is 0, but 0 * inf = nan, so we handle it
    # Actually, 0 * (1/d) is 0. But if d=0, 1/d is inf. 
    # We filled diagonal of inv_dist with 0, so diagonal is safe.
    # For off-diagonal, d > 0.
    penalty_cum = lambda_cum * usage * inv_dist
    
    # 2. Immediate penalty for the current tour
    # This ensures we don't just stay in the same local optimum
    lambda_curr = 0.05
    penalty_curr = np.zeros_like(edge_distance)
    
    # Extract edges from local_opt_tour
    # local_opt_tour is likely a list of nodes [0, 1, ..., n-1]
    if local_opt_tour is not None and len(local_opt_tour) == n:
        for i in range(n):
            u = local_opt_tour[i]
            v = local_opt_tour[(i + 1) % n]
            penalty_curr[u, v] = lambda_curr * inv_dist[u, v]
            penalty_curr[v, u] = lambda_curr * inv_dist[v, u]
    
    # Apply penalties
    penalized = penalized + penalty_cum + penalty_curr
    
    # Ensure the diagonal remains 0
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure symmetry
    penalized = 0.5 * (penalized + penalized.T)
    
    # Ensure all values are finite and non-negative
    penalized = np.where(np.isfinite(penalized), penalized, 0.0)
    penalized = np.maximum(penalized, 0.0)
    
    return penalized
# EVOLVE-BLOCK-END
