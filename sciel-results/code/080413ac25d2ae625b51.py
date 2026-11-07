"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute a heuristic attractiveness matrix for ACO on CVRP.
    
    The heuristic is computed once per instance before the colony runs.
    We want to:
    1. Strongly favor short edges (use 1/d^gamma with gamma > 1)
    2. Encourage ants to stay on routes longer (penalize early return to depot)
    3. Consider demand: edges to high-demand customers might be slightly less
       attractive if they fill up the truck quickly, but this is complex.
       Instead, we use a combination of distance-based heuristic with a 
       depot-return penalty.
    
    Key insight: The heuristic should guide ants toward dense clusters of 
    customers to minimize total distance. We use 1/d^2.5 as base, then 
    penalize edges back to depot for non-depot nodes to encourage fuller loads.
    """
    # Ensure we work with float arrays
    d = np.asarray(distance_matrix, dtype=np.float64)
    
    # Base heuristic: inverse distance with exponent 2.5
    # Add small epsilon to avoid division by zero
    epsilon = 1e-9
    eta = 1.0 / ((d + epsilon) ** 2.5)
    
    # Penalize returning to depot (column 0) for non-depot nodes (i != 0)
    # This encourages ants to visit more customers before returning,
    # leading to fuller trucks and fewer trips
    if d.shape[0] > 1:
        eta[1:, 0] *= 0.6
    
    # Also, we can slightly boost edges from depot to customers (row 0)
    # to help ants start their routes well
    if d.shape[1] > 1:
        eta[0, 1:] *= 1.2
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
