import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Distance from each node to depot (node 0)
    # Assuming symmetric matrix, dist[j,0] == dist[0,j]. 
    # The constraint uses d[j,0].
    d_to_depot = distance_matrix[:, 0]
    
    # Numerator: prize of destination node j
    # Shape (1, n)
    num = prizes[None, :] + 1e-9
    
    # Denominator: d[i,j] + d[j,0]
    # distance_matrix is (n, n)
    # d_to_depot[None, :] is (1, n)
    # Broadcasting: d[i,j] + d[j,0]
    denom = distance_matrix + d_to_depot[None, :] + 1e-9
    
    eta = num / denom
    
    # Sanitize
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
