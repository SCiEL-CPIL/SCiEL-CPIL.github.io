"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for ACO orienteering.
    
    Structural Change: 
    Replaces row-normalized rank-based feasibility with a Global Efficiency 
    times Accurate Slack model.
    
    Heuristic eta[i, j] = (Prize_Efficiency[j]) * (Slack[i, j])^gamma
    
    Where:
    - Prize_Efficiency[j] = prize[j] / (d[j, 0] + eps)
      Represents the value density of node j relative to its return cost.
    - Slack[i, j] = max_len - d[0, i] - d[i, j] - d[j, 0]
      Represents the remaining path length budget if the ant moves i->j and then returns to 0.
      This accounts for the distance already traveled to i (lower bound d[0,i]).
    - gamma: Exponent to sharpen preference for moves with more remaining slack.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
        
    eps = 1e-9
    
    # Precompute distances from origin (node 0)
    d_from_0 = distance_matrix[0, :]  # shape (n,)
    d_to_0 = distance_matrix[:, 0]     # shape (n,)
    
    # 1. Compute Global Node Efficiency
    # Efficiency of node j: How much prize per unit of return distance.
    # We add epsilon to denominator to avoid division by zero.
    # We clip prizes to be non-negative (though they are [0,1]).
    prizes_safe = np.maximum(prizes, 0.0)
    return_cost_safe = d_to_0 + eps
    node_efficiency = prizes_safe / return_cost_safe  # shape (n,)
    
    # 2. Compute Accurate Slack Matrix
    # slack[i, j] = max_len - (d[0, i] + d[i, j] + d[j, 0])
    # d[0, i] is the minimum distance to reach i.
    # d[i, j] is the step cost.
    # d[j, 0] is the return cost.
    
    # d_0i_matrix: (n, 1) for broadcasting against rows i
    d_0i_matrix = d_from_0[:, np.newaxis]
    # d_j0_matrix: (1, n) for broadcasting against columns j
    d_j0_matrix = d_to_0[np.newaxis, :]
    
    # Total cost of path 0 -> i -> j -> 0
    total_path_cost = d_0i_matrix + distance_matrix + d_j0_matrix
    
    slack = max_len - total_path_cost
    
    # Clip negative slack to 0. Moves with negative slack are infeasible 
    # (or leave no room for return), so they should have 0 attractiveness.
    slack = np.maximum(slack, 0.0)
    
    # 3. Combine Efficiency and Slack
    # We want to encourage moves that are both efficient (high node_efficiency) 
    # and leave plenty of budget (high slack).
    
    # Apply a power to slack to control sensitivity. 
    # gamma=1.0 is linear. gamma=2.0 is quadratic.
    # Higher gamma makes ants prefer moves that leave *much* more budget, 
    # potentially avoiding tight fits that might trap them later.
    gamma = 1.5
    
    slack_term = np.power(slack, gamma)
    
    # eta[i, j] = node_efficiency[j] * slack_term[i, j]
    # Broadcast node_efficiency (n,) to (n, n) along rows
    heuristic = node_efficiency[np.newaxis, :] * slack_term
    
    # 4. Normalization and Safety
    # ACO probabilities are proportional to tau * eta. 
    # If eta values are too large or small, it can cause numerical issues or 
    # make pheromone updates dominate.
    # We normalize each row to have a mean of 1 (or similar) to keep scale consistent.
    
    # Check for time
    if time.time() > deadline - 1.0:
         # Fallback: return a simple valid matrix
         return np.ones((n, n)) * 1e-9

    # Handle rows that might be all zeros (if no moves are feasible from i)
    row_sums = np.sum(heuristic, axis=1, keepdims=True)
    row_sums_safe = np.maximum(row_sums, eps)
    
    # Normalize by row sum to keep magnitude stable
    heuristic = heuristic / row_sums_safe
    
    # Set diagonal to minimal value (self-loop not allowed/needed)
    np.fill_diagonal(heuristic, 1e-9)
    
    # Ensure all values are positive and finite
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
