"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    eps = 1e-9
    
    # Structural change: rank-based accessibility heuristic
    # Instead of a continuous prize/cost ratio, use normalized ranks
    # to create a more robust, bounded heuristic that captures
    # both prize quality and path feasibility.
    
    d_j0 = distance_matrix[:, 0]  # distance from each node to origin
    marginal_cost = distance_matrix + d_j0[None, :]  # C[i,j] = d[i,j] + d[j,0]
    
    # Normalize prizes to [0, 1] range (they're already in [0,1] but be safe)
    prize_max = np.max(prizes)
    if prize_max > 0:
        prize_norm = prizes / prize_max
    else:
        prize_norm = prizes.copy()
    
    # Normalize marginal costs per row to [0, 1]
    # Lower cost = higher feasibility score
    row_max_cost = np.max(marginal_cost, axis=1, keepdims=True)
    row_min_cost = np.min(marginal_cost, axis=1, keepdims=True)
    cost_range = row_max_cost - row_min_cost
    cost_range = np.where(cost_range < eps, eps, cost_range)
    
    # Feasibility score: 1 for cheapest, 0 for most expensive (per row)
    feasibility = 1.0 - (marginal_cost - row_min_cost) / cost_range
    feasibility = np.clip(feasibility, 0.0, 1.0)
    
    # Prize score: emphasize top prizes with a power
    prize_score = np.power(prize_norm[None, :], 3.0)  # cubic to sharpen
    
    # Combine: product of prize score and feasibility, then amplify
    # This creates a bounded [0,1] heuristic that's structured differently
    # from a ratio - it's a multiplicative gate.
    heuristic = prize_score * feasibility
    
    # Amplify the spread: scale so that the max is around 1.0
    # and use a power to create steeper preference for best options
    h_max = np.max(heuristic)
    if h_max > eps:
        heuristic = heuristic / h_max
    
    # Apply a power to sharpen: values near 1 stay high, lower values drop
    heuristic = np.power(heuristic, 2.0)
    
    # Ensure all values are positive
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Set diagonal to minimal value
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
