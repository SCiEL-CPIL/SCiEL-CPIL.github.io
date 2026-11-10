"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    Uses a multiplicative cost model: penalizes edges where either the travel
    distance or the return distance is large, more strongly than an additive
    model. This creates a sharper preference for 'efficient' local moves that
    are both close to the current node and close to the start.
    
    Formula: eta[i, j] = prize[j] / (distance[i, j] * distance[j, 0] + epsilon)
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Ensure shapes are compatible
    n = prizes.shape[0]
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-9
    
    # distance[j, 0] is the cost to return to start from j
    return_cost = distance_matrix[:, 0]  # Shape (n,)
    
    # Multiplicative cost: product of travel distance and return distance
    # This penalizes long returns heavily, even if travel is short, and vice versa.
    # It effectively creates a 'potential well' around nodes that are close to both i and 0.
    multiplicative_cost = distance_matrix * return_cost[None, :] + epsilon
    
    # Heuristic: Prize / Multiplicative Cost
    heuristic = prizes[None, :] / multiplicative_cost
    
    # Handle any non-finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
