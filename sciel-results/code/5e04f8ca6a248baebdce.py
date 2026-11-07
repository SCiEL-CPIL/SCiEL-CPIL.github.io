import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # d[i, j] is distance from i to j
    # d[j, 0] is distance from j to depot
    
    # We want to reward nodes that are:
    # 1. High prize
    # 2. Close to current node i (low d[i, j])
    # 3. Close to depot (low d[j, 0]) because this preserves budget
    
    # Heuristic: prize[j] / (d[i, j] * (d[j, 0] + eps))
    
    prize_col = prizes[None, :]  # 1 x n
    dist_ij = distance_matrix + eps  # n x n
    dist_j0 = distance_matrix[:, 0:1] + eps  # n x 1, broadcast to n x n
    
    # The term d[j, 0] is the same for all i in a given column j? 
    # No, distance_matrix[:, 0:1] is the first column, which is d[i, 0].
    # We need d[j, 0]. That is distance_matrix[:, 0].
    # Let's be careful with indices.
    # distance_matrix[i, j] is d(i, j).
    # We need d(j, 0). That is distance_matrix[j, 0].
    # In matrix form, column 0 of distance_matrix is d(0, 0), d(1, 0), ..., d(n-1, 0).
    # So dist_to_depot = distance_matrix[:, 0] is a vector of length n.
    # We need to broadcast this to n x n such that entry (i, j) uses dist_to_depot[j].
    
    dist_to_depot = distance_matrix[:, 0] + eps  # 1D array of length n
    # Reshape to (1, n) to broadcast across rows i
    dist_to_depot_row = dist_to_depot[None, :]  # 1 x n
    
    # heuristic[i, j] = prize[j] / (d[i, j] * d[j, 0])
    heuristic = prize_col / (dist_ij * dist_to_depot_row)
    
    # Ensure positive and finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
