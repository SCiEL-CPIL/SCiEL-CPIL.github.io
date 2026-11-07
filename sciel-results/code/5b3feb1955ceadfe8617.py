import time
import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    start_time = time.time()
    deadline = start_time + 10.0
    
    n = distance_matrix.shape[0]
    
    # 1. Compute Savings
    # S_ij = d_i0 + d_0j - d_ij
    d_i0 = distance_matrix[:, 0].reshape(-1, 1) # (n, 1)
    d_0j = distance_matrix[0, :].reshape(1, -1) # (1, n)
    S = d_i0 + d_0j - distance_matrix
    
    # Normalize Savings
    S_max = np.max(S)
    if S_max > 0:
        S_norm = S / S_max
    else:
        S_norm = np.zeros_like(S)
    
    # 2. Construct Heuristic
    # Use 1/d (alpha=1) to reduce penalty on long edges
    # Increase weight of savings (lambda_s) to emphasize "flow" through high-savings edges
    alpha = 1.0
    lambda_s = 5.0
    
    d_safe = distance_matrix + 1e-9
    dist_term = 1.0 / (d_safe ** alpha)
    multiplier = 1.0 + lambda_s * S_norm
    eta = dist_term * multiplier
    
    # Ensure positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # Time guard check
    if time.time() > deadline:
        return 1.0 / (distance_matrix + 1e-9)
        
    return eta
