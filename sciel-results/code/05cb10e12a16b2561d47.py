"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ACO of CVRP.
    Larger values are more attractive.
    
    Strategy: Combine inverse distance (shorter is better) with a demand-aware
    component that rewards edges leading to nodes with lower demand (leaves more
    capacity for future stops) and a coordinate-based component that rewards
    moving toward the depot (to facilitate return) or away from already-visited
    (but we don't know visited here, so we use a static proxy: proximity to depot
    for high-demand nodes, or a balance).
    
    Actually, a simpler and effective heuristic for CVRP in ACO:
    eta[i,j] = 1 / (distance[i,j] * (1 + demands[j]))
    
    This penalizes long distances and high-demand customers, encouraging the ant
    to pick close, low-demand customers first, which often leads to better packing.
    """
    # Ensure demands is a numpy array
    demands = np.asarray(demands, dtype=np.float64)
    
    # Base heuristic: inverse distance
    eta = 1.0 / (distance_matrix + 1e-9)
    
    # Demand-aware adjustment: penalize edges to high-demand customers
    # We want to visit low-demand customers more often to allow more customers per route
    # Multiply by 1/(1 + demand) to reduce attractiveness of high-demand nodes
    # This is a static heuristic, so it's the same for all ants
    demand_penalty = 1.0 / (1.0 + demands)
    # For each column j, scale by demand_penalty[j]
    eta = eta * demand_penalty[np.newaxis, :]
    
    # Ensure no zero or negative values
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
