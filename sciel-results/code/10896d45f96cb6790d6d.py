"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for Ant Colony Optimization.
    
    Uses a 'Regret-Amplified Efficiency' heuristic:
    eta_ij = ( (prize_j + eps) / (d_ij + d_j0 + eps) )^gamma
    
    This accounts for the full detour cost of visiting node j and returning to start,
    which is critical for the length constraint. The exponent gamma amplifies the
    difference between high-efficiency and low-efficiency edges.
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Ensure shapes are correct and types are float
    if prizes.shape != (distance_matrix.shape[0],):
        raise ValueError("Shape mismatch between prizes and distance_matrix")
        
    n = distance_matrix.shape[0]
    if distance_matrix.shape != (n, n):
        raise ValueError("distance_matrix must be square")
        
    eps = 1e-9
    gamma = 2.5  # Exponent to sharpen the heuristic preference
    
    # Vectorized calculation
    # prize_j is broadcast across rows i
    # d_ij is the distance matrix
    # d_j0 is the distance from each node j to node 0 (column 0 of distance_matrix)
    
    prize_j = prizes[None, :] + eps
    d_j0 = distance_matrix[:, 0]  # Shape (n,)
    # d_j0 needs to be broadcast across rows i, so shape (1, n)
    d_j0_row = d_j0[None, :]
    
    # Total cost of visiting j from i and returning to 0
    # Note: The constraint is d[i,j] + d[j,0]. 
    # distance_matrix[i,j] is d[i,j].
    # distance_matrix[j,0] is d[j,0].
    # So we need distance_matrix[i,j] + distance_matrix[j,0].
    
    # distance_matrix is (n, n). 
    # We want for each i, j: distance_matrix[i, j] + distance_matrix[j, 0].
    # distance_matrix[j, 0] is the j-th element of the vector distance_matrix[:, 0].
    # So we add distance_matrix[:, 0][None, :] to distance_matrix.
    
    cost = distance_matrix + d_j0_row + eps
    
    # Calculate efficiency ratio
    ratio = prize_j / cost
    
    # Apply exponent
    eta = np.power(ratio, gamma)
    
    # Ensure non-finite and non-positive values are handled
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, eps)
    
    return eta
# EVOLVE-BLOCK-END
