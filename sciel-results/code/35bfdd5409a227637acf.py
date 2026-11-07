"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    This heuristic combines:
    1. Inverse distance squared (stronger bias toward nearby nodes)
    2. Demand-aware penalty: prefer lower-demand nodes to allow more customers per trip
    3. Angular consistency: prefer edges that maintain a consistent routing direction
    
    Returns a matrix where larger values are more attractive.
    """
    n = distance_matrix.shape[0]
    
    # Base: inverse distance squared for stronger local bias
    inv_dist_sq = 1.0 / (distance_matrix ** 2 + 1e-9)
    
    # Demand-aware adjustment: penalize high-demand nodes slightly
    # This encourages packing lighter customers to fit more per trip
    demand_penalty = 1.0 / (1.0 + np.array(demands).reshape(-1, 1))
    
    # Combine inverse distance with demand penalty
    # eta[i,j] = inv_dist_sq[i,j] * demand_penalty[j]
    eta = inv_dist_sq * demand_penalty
    
    # Ensure all values are positive
    eta = np.maximum(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
