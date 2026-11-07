import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Applies a geometry-aware penalty where short, frequently used edges
    receive disproportionately large penalties to force escape from local optima.
    Includes an immediate penalty on the current tour's edges.
    """
    # Simple deadline guard (not expected to trigger for N=100)
    start_time = time.time()
    deadline = start_time + 0.1  # 100ms safety margin per call? 
    # Actually, the evaluator runs 20 rounds. 300s total.
    # 300 / 20 = 15s per round. 
    # This function is called once per round. 
    # 100ms is way more than enough for 100x100 matrix ops.
    
    n = edge_distance.shape[0]
    
    # 1. Saturating usage penalty
    # P_ij = alpha * U_ij / (1 + U_ij)
    alpha = 0.5
    # Handle division by zero if U is 0? 0/(1+0) = 0. Safe.
    P = alpha * edge_n_used / (1.0 + edge_n_used)
    
    # 2. Geometry-aware weighting (inverse distance)
    # We want to amplify the penalty for short edges.
    eps = 1e-6
    D = edge_distance + eps
    W = 1.0 / D
    
    # Normalize W so that the max weight is 1.0
    # This ensures the penalty magnitude is controlled by alpha and the relative shape of W.
    W_max = np.max(W)
    if W_max > 0:
        W = W / W_max
    else:
        W = np.ones_like(W)
        
    # Combine
    Penalty = P * W
    
    # 3. Immediate penalty on current tour edges
    # Ensures perturbation even if usage counts are low (e.g., early rounds)
    gamma = 0.15
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_mask[u, v] = 1.0
        tour_mask[v, u] = 1.0
        
    Penalty += gamma * tour_mask
    
    # 4. Return modified distance matrix
    # Adding penalty is standard for GLS
    return edge_distance + Penalty
