"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a demand-aware heuristic matrix for ACO.
    
    The heuristic combines inverse distance with a penalty for high demand customers.
    This encourages ants to construct routes that balance load and reduce the frequency
    of returns to the depot, which typically lowers total travel distance in CVRP.
    
    Args:
        distance_matrix: NxN matrix of Euclidean distances.
        coordinates: Nx2 array of coordinates (unused in this specific heuristic but kept for signature).
        demands: Array of length N with customer demands (floats).
        capacity: Vehicle capacity (unused in this specific heuristic but kept for signature).
        
    Returns:
        NxN matrix of edge attractiveness (larger is more attractive).
    """
    # Ensure inputs are numpy arrays for efficient operations
    d_mat = np.asarray(distance_matrix, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # 1. Base heuristic: Inverse distance
    # Add a small epsilon to prevent division by zero for diagonal elements (distance 0)
    # Although diagonal elements are typically ignored in ACO construction,
    # it's good practice to ensure finite values.
    epsilon = 1e-9
    base_heuristic = 1.0 / (d_mat + epsilon)
    
    # 2. Demand-aware penalty
    # Penalize edges leading to high-demand customers.
    # The penalty factor is 1 / (1 + alpha * demand).
    # alpha = 0.5 provides a moderate penalty for demands 1-9.
    # For demand 1: factor = 1/1.5 = 0.66
    # For demand 9: factor = 1/5.5 = 0.18
    # This significantly reduces the attractiveness of high-demand nodes,
    # encouraging ants to group smaller demands or plan routes more carefully.
    alpha = 0.5
    demand_penalty = 1.0 / (1.0 + alpha * dem)
    
    # Apply penalty to each column j (destination node)
    # demand_penalty is shape (N,), base_heuristic is (N, N)
    # We want to multiply each column j by demand_penalty[j]
    heuristic = base_heuristic * demand_penalty[np.newaxis, :]
    
    # 3. Ensure all values are positive and finite
    # The contract states non-positive values become 1e-9.
    # We also check for non-finite values just in case of numerical issues.
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
