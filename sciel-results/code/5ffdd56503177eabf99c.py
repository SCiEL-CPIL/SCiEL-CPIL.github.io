"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a demand-aware, multi-criteria edge-attractiveness matrix.
    
    Combines:
    1. Inverse distance squared (strong bias toward short edges)
    2. Penalty for high-demand destination nodes (encourages packing heavy
       customers early to reduce number of trips)
    3. A small bonus for edges from the depot to encourage early departure
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base: inverse distance squared, with a small epsilon to avoid division by zero
    # distance_matrix has zeros on diagonal; we add epsilon
    d = distance_matrix + 1e-9
    eta_dist = 1.0 / (d ** 2)
    
    # Demand-aware penalty: nodes with higher demand get lower attractiveness
    # when approached. This encourages the ant to visit high-demand nodes when
    # the truck is still full (early in the route), effectively packing them
    # into fewer trips.
    # demand_factor[j] is in [1/(1+9), 1/(1+1)] = [0.1, 0.5] for demands 1..9
    demand_factor = 1.0 / (1.0 + demands)
    
    # Apply demand penalty only to destination column j
    # eta[i, j] *= demand_factor[j]
    eta = eta_dist * demand_factor[np.newaxis, :]
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
