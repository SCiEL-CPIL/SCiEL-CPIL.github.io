"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for the Ant Colony Optimization.
    
    The heuristic for moving from node i to node j is defined as the prize of node j
    divided by the sum of the distance from i to j and the distance from j to the origin (0).
    This encourages moving to high-prize nodes that are close to the current position
    and also close to the origin, ensuring the ant can return to the start within the
    length constraint.
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # d[j, 0] is the distance from node j to the origin.
    # distance_matrix[:, 0] gives a column vector of distances to node 0.
    # We need to add this to each row i of the distance matrix.
    # distance_matrix[i, j] + distance_matrix[j, 0]
    
    # To do this efficiently:
    # dist_to_origin = distance_matrix[:, 0]  # shape (N, 1)
    # denom = distance_matrix + dist_to_origin  # broadcasting adds dist_to_origin to each row?
    # No, distance_matrix is (N, N). dist_to_origin is (N, 1).
    # distance_matrix[i, j] + dist_to_origin[j, 0] requires dist_to_origin to be (1, N) to broadcast along rows?
    # If dist_to_origin is (N, 1), then distance_matrix[i, j] + dist_to_origin[i, 0] would add the distance of i to 0.
    # We want to add distance of j to 0. So we need a vector of shape (1, N) containing d[0,0], d[1,0], ..., d[N-1,0].
    # distance_matrix[:, 0] is (N, 1). We need distance_matrix[0, :]? No, distance is symmetric, so d[j,0] == d[0,j].
    # So d[j,0] is the j-th element of the first row (or column).
    # d_to_0 = distance_matrix[0, :]  # shape (N,)
    # denom = distance_matrix + d_to_0[None, :]  # shape (N, N)
    
    d_to_0 = distance_matrix[0, :]
    denom = distance_matrix + d_to_0[None, :]
    
    # Avoid division by zero
    denom = np.where(denom == 0, 1e-9, denom)
    
    # Heuristic: Prize of destination / (Cost of moving there + Cost of returning from there)
    # prizes[None, :] is (1, N), broadcasts to (N, N)
    eta = prizes[None, :] / denom
    
    # Ensure positivity for ACO (non-positive becomes 1e-9 in evaluator, but we can ensure it here)
    eta = np.where(eta > 0, eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
