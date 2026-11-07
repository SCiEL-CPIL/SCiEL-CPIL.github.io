import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization.
    
    This implementation uses a 'Regret-Amplified Efficiency' heuristic.
    Instead of a simple prize/distance ratio, we use:
    eta_ij = ( prize_j / (d_ij + d_j0 + eps) )^gamma
    
    The addition of d_j0 (return cost) accounts for the full detour cost of visiting node j,
    which is critical for the length constraint.
    The exponent gamma > 1 amplifies the difference between high-efficiency and low-efficiency
    edges, acting as a static 'regret' mechanism that strongly biases ants towards the
    most efficient moves and away from suboptimal ones.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = len(prizes)
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure positive values to avoid division by zero or negative logs
    eps = 1e-9
    
    # d_j0 is the distance from node j back to start (node 0)
    # distance_matrix[j, 0]
    d_j0 = distance_matrix[:, 0]
    
    # Cost of visiting j and returning: d_ij + d_j0
    # We create a matrix of these costs
    # d_ij is distance_matrix
    # d_j0 needs to be broadcasted to columns
    costs = distance_matrix + d_j0[None, :] + eps
    
    # Base efficiency: Prize_j / Cost
    # prizes[None, :] broadcasts prizes to rows
    efficiency = (prizes[None, :] + eps) / costs
    
    # Amplify the efficiency to create a stronger bias (Regret-like effect)
    # Higher gamma makes the heuristic more selective
    gamma = 2.0
    heuristic = np.power(efficiency, gamma)
    
    # Replace non-positive or non-finite values with a small constant
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Check time
    if time.time() > deadline:
         # Return a simple fallback if we are running out of time (unlikely for this small computation)
         return (prizes[None, :] + 1e-9) / (distance_matrix + 1e-9)

    return heuristic
