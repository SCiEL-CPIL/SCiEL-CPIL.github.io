"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for CVRP using a savings-based heuristic.
    
    The heuristic combines inverse distance with a savings efficiency factor.
    Savings s_ij = d_i0 + d_0j - d_ij.
    Efficiency = s_ij / (d_i0 + d_0j).
    Heuristic = (1 / (d_ij + eps)) * (1 + Efficiency)
    """
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    # Get distances from depot (node 0)
    d_depot = distance_matrix[0, :]
    
    # Calculate savings matrix: S[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # d[i, 0] is distance from i to depot
    # d[0, j] is distance from depot to j
    d_i_depot = d_depot[:, np.newaxis] # Column vector
    d_depot_j = d_depot[np.newaxis, :] # Row vector
    
    savings = d_i_depot + d_depot_j - distance_matrix
    
    # Ensure savings are non-negative (should be due to triangle inequality, but clamp for safety)
    savings = np.maximum(savings, 0.0)
    
    # Calculate efficiency: s_ij / (d_i0 + d_0j)
    # Avoid division by zero if depot distance is 0 (i=0 or j=0)
    denom = d_i_depot + d_depot_j
    denom_safe = np.where(denom > 1e-9, denom, 1.0)
    
    efficiency = savings / denom_safe
    
    # Heuristic: Inverse distance boosted by efficiency
    # Base heuristic is 1/d_ij
    eps = 1e-9
    base_heur = 1.0 / (distance_matrix + eps)
    
    # Boost factor: 1 + efficiency (ranges from 1 to 2)
    boost = 1.0 + efficiency
    
    heur_matrix = base_heur * boost
    
    # Handle the depot row/col? 
    # The problem says "Node 0 is the depot". 
    # Ants start at depot. They pick unvisited customers.
    # The heuristic is used for transitions i -> j.
    # If i is depot, j is customer. 
    # If i is customer, j is customer or depot.
    # The matrix needs to be valid for all i, j.
    
    # Ensure non-finite values are handled
    heur_matrix = np.where(np.isfinite(heur_matrix), heur_matrix, 1e-9)
    
    # Heuristic entries that are not positive become 1e-9
    heur_matrix = np.where(heur_matrix > 0, heur_matrix, 1e-9)
    
    return heur_matrix
# EVOLVE-BLOCK-END
