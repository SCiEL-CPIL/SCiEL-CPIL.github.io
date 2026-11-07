"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    
    The heuristic is designed to favor nodes that have high prizes and low
    total cost to visit and return to the origin. The cost of moving from
    current node i to candidate node j and then returning to origin 0 is
    approximated by d[i, j] + d[j, 0].
    
    eta[i, j] = (prize[j] + epsilon) / (d[i, j] + d[j, 0] + epsilon)
    """
    # Get dimensions
    n = len(prizes)
    
    # Epsilon for numerical stability
    epsilon = 1e-9
    
    # Vector of prizes
    # prizes shape: (n,)
    # We need prize[j] for all i, j. So we broadcast prizes to (1, n)
    prize_col = prizes[None, :] + epsilon
    
    # Distance matrix d[i, j]
    # d[i, j] shape: (n, n)
    
    # Distance to origin d[j, 0]
    # This is the 0-th column of distance_matrix
    # d[j, 0] is distance_matrix[j, 0]
    # We need this for each column j in the matrix eta[i, j].
    # So we want a row vector of d[j, 0] for j in 0..n-1.
    # distance_matrix[:, 0] has shape (n,).
    # We broadcast it to (1, n) to add to distance_matrix (n, n).
    dist_to_origin = distance_matrix[:, 0][None, :] + epsilon
    
    # Calculate the combined cost in the denominator
    # cost[i, j] = d[i, j] + d[j, 0]
    cost_matrix = distance_matrix + dist_to_origin
    
    # Calculate heuristic
    # eta[i, j] = prize[j] / cost[i, j]
    eta = prize_col / cost_matrix
    
    # Ensure non-positive values are replaced by 1e-9
    # Although with epsilon in numerator and denominator, values should be > 0.
    # But if prize is 0, numerator is epsilon. Denominator is > 0. Result > 0.
    # The spec says "Heuristic entries that are not positive become 1e-9".
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
