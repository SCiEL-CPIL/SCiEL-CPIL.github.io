"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for Ant Colony Optimization.
    
    The heuristic eta[i, j] is defined as the prize of node j divided by the 
    sum of the distance from i to j and the distance from j to the start node (0).
    This accounts for the cost of returning to the start, which is critical 
    for the orienteering problem's length constraint.
    
    A power of 1.5 is applied to sharpen the preference for high-value moves.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    
    # Distance from each node j to the start node (node 0)
    # Shape: (n,)
    dist_to_start = distance_matrix[:, 0]
    
    # We want a matrix H where H[i, j] = (prize[j] / (d[i, j] + d[j, 0]))^alpha
    # 
    # d[i, j] is distance_matrix
    # d[j, 0] is dist_to_start[j]
    #
    # To vectorize:
    # Denominator[i, j] = distance_matrix[i, j] + dist_to_start[j]
    
    # Create a column vector of dist_to_start to add to each row of distance_matrix
    # dist_to_start is (n,), we need (1, n) to broadcast against (n, n)
    denom = distance_matrix + dist_to_start[None, :]
    
    # Avoid division by zero
    denom = np.maximum(denom, 1e-9)
    
    # Numerator is prize[j] broadcast to (n, n)
    # prize is (n,), we need (1, n)
    numer = prizes[None, :]
    
    # Calculate base ratio
    ratio = numer / denom
    
    # Ensure positive values for log/power operations, though ratio is already positive
    # Apply power to sharpen heuristic
    alpha = 1.5
    H = np.power(ratio, alpha)
    
    # Ensure all values are positive (as per task: "Heuristic entries that are not positive become 1e-9")
    # Since prizes can be 0 (at node 0), H[i, 0] will be 0.
    # The task says "Heuristic entries that are not positive become 1e-9".
    # 0 is not positive. So we should replace 0 (and any negative, though none expected) with 1e-9.
    H = np.where(H > 0, H, 1e-9)
    
    # The diagonal H[i, i] is technically for moving to self, which isn't allowed in the loop 
    # (unvisited check), but let's keep it consistent.
    # prize[0] is 0, so H[i, 0] is 0 -> 1e-9. This is correct, moving to start is not useful for prize collection 
    # unless it's the end, but the heuristic guides *moves to unvisited nodes*.
    
    return H
# EVOLVE-BLOCK-END
