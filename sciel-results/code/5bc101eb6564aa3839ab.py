import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    # prizes: shape (N,)
    # distance_matrix: shape (N, N)
    
    # Base heuristic: prize / distance
    # Add epsilon to distance to avoid division by zero
    base = prizes[None, :] / (distance_matrix + 1e-9)
    
    # Slack calculation
    # d[i, j] is distance_matrix[i, j]
    # d[j, 0] is distance_matrix[j, 0]
    # We need d[j, 0] for each column j.
    # distance_matrix[:, 0] gives a vector [d[0,0], d[1,0], ..., d[N-1,0]]
    # We want this to be broadcasted across rows.
    return_costs = distance_matrix[:, 0]  # shape (N,)
    return_costs_row = return_costs[None, :]  # shape (1, N)
    
    slack = max_len - distance_matrix - return_costs_row
    
    # Combine
    heuristic = base * slack
    
    # Enforce positivity
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    return heuristic
