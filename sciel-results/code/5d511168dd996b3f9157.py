import numpy as np
def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    # d_j0 is distance from each node j to node 0
    d_j0 = distance_matrix[:, 0]
    
    # Heuristic: Prize / (Distance from i to j + Weighted Distance from j to 0)
    # Using weight 1.0 for d_j0 to strongly penalize long returns
    # Adding epsilon to avoid division by zero
    eps = 1e-9
    
    # Shape: (n, 1) for broadcasting against (n, n)
    # distance_matrix[i, j] is d_ij
    # d_j0[j] is d_j0
    
    # We want H[i, j]
    # Denominator: d_ij + d_j0
    # d_ij is (n, n)
    # d_j0 is (n,) -> broadcast to (1, n)
    
    denom = distance_matrix + d_j0[None, :] + eps
    numer = prizes[None, :] + eps
    
    H = numer / denom
    
    # Ensure positive
    H[H <= 0] = 1e-9
    
    return H
