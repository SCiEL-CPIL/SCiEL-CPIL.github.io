# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix for ant colony orienteering.
    
    Strategy: Prize-to-Commitment-Cost Ratio.
    The heuristic for edge (i, j) is the prize of node j divided by the 
    squared sum of the distance to j and the distance from j back to the start.
    This encourages moves to high-prize nodes that are close and have a short 
    return trip, which is crucial for staying within the max_len constraint.
    """
    n = len(prizes)
    eps = 1e-9
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Calculate the commitment cost for each edge (i, j)
    # Cost = d[i, j] + d[j, 0]
    # d[j, 0] is the distance from node j to the start node 0.
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    
    # commitment_cost[i, j] = distance_matrix[i, j] + d_j0[j]
    commitment_cost = distance_matrix + d_j0[None, :]
    
    # Ensure commitment cost is positive to avoid division by zero
    commitment_cost = np.maximum(commitment_cost, eps)
    
    # Heuristic = (Prize[j] / commitment_cost[i, j])^2
    # Using square to strongly favor efficient moves
    prize_component = prizes[None, :]  # shape (1, n)
    
    heuristic = (prize_component / commitment_cost) ** 2
    
    # Handle node 0 (prize 0) and any numerical issues
    # If prize is 0, heuristic is 0. We need positive values.
    # The problem states "Heuristic entries that are not positive become 1e-9".
    # So we can just clamp.
    heuristic = np.where(heuristic > 0, heuristic, eps)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    # Normalize to mean 1 to keep pheromone dynamics stable
    mean_val = np.mean(heuristic)
    if mean_val > eps:
        heuristic = heuristic / mean_val
        
    return heuristic
# EVOLVE-BLOCK-END
