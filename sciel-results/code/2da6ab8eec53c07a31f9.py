import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    eps = 1e-9
    
    # Distance from each node to origin (node 0)
    dist_to_origin = distance_matrix[:, 0].copy()
    
    # Prize matrix: prize[j] for each target node j
    prize_col = prizes[None, :].copy()
    
    # Heuristic: multiplicative penalty on both d[i,j] and d[j,0]
    # This creates hyperbolic decay in both dimensions,
    # strongly favoring nodes close to both current position and origin
    # Square the prize to emphasize high-value nodes
    d_ij = distance_matrix + eps
    d_j0 = dist_to_origin[None, :] + eps
    
    eta = (prize_col ** 2) / (d_ij * d_j0)
    
    # Ensure all entries are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
