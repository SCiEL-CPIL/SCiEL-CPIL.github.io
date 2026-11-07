"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes a static heuristic matrix for CVRP.
    
    The heuristic combines distance and demand information.
    It encourages short edges and edges leading to nodes with higher demand
    (to be visited early when capacity is available).
    
    H_ij = (1 / (d_ij + eps)) * (base + demand_j / capacity)
    """
    eps = 1e-9
    # Ensure demands are float for division
    d = np.asarray(demands, dtype=float)
    cap = float(capacity)
    
    # Inverse distance component
    # d_ij is the distance from i to j
    inv_dist = 1.0 / (distance_matrix + eps)
    
    # Demand component
    # We want to favor visiting nodes with higher demand earlier.
    # So the attractiveness of moving TO node j should increase with d_j.
    # Normalize demand to [0, 1] range roughly, or just use ratio.
    # If d_j is large, it's "more attractive" to pick it now.
    demand_factor = 1.0 + (d / cap)
    
    # Combine them
    # We weight the distance heavily, but modulate by demand.
    heuristic = inv_dist * demand_factor
    
    # Ensure no zeros or negatives (though inputs should be positive)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
