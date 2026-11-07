"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute a heuristic matrix for Ant Colony Optimization.
    
    Strategy: Maximum Coverage Set Selection Heuristic
    Instead of modeling the heuristic as a local potential field dependent on the 
    current node i (which encourages local path optimization), we model the 
    orienteering problem as a global set selection problem. 
    
    The attractiveness of moving to node j is determined solely by the global 
    efficiency of node j: its prize relative to the cost of committing to it 
    (distance from origin). This makes the heuristic independent of the current 
    node i, encouraging ants to converge on the globally most efficient prize 
    clusters within the return-to-origin budget.
    
    Specifically:
    eta[i, j] = prize[j] / (d[j, 0] + epsilon)
    
    This ignores the local transition cost d[i, j] in the heuristic, relying on 
    the pheromone trail to handle local connectivity costs, while the heuristic 
    strongly biases the ants toward high-value, low-return-cost nodes.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure non-negative and finite
    prizes = np.clip(prizes, 0, None)
    distance_matrix = np.clip(distance_matrix, 0, None)
    
    n = len(prizes)
    eps = 1e-9
    
    # Calculate the global efficiency of each node j
    # efficiency[j] = prize[j] / (d[j, 0] + eps)
    return_costs = distance_matrix[:, 0]
    efficiency = prizes / (return_costs + eps)
    
    # Construct the heuristic matrix h
    # h[i, j] = efficiency[j] for all i
    # This creates a matrix where each row is identical, containing the efficiency scores.
    h = np.tile(efficiency, (n, 1))
    
    # Ensure all values are positive and finite
    # If a node has 0 prize or infinite distance, efficiency is 0 or 0, which is handled by max
    # If a node is the origin (j=0), prize is 0, so efficiency is 0.
    # The problem states prize 0 at node 0.
    # We replace non-positive or non-finite values with 1e-9.
    h = np.where(np.isfinite(h), h, 1e-9)
    h = np.maximum(h, 1e-9)
    
    # Time check
    if time.time() > deadline:
        # If we're running out of time, return the current result
        # This should not happen given the vectorized operations
        return h
    
    return h
# EVOLVE-BLOCK-END
