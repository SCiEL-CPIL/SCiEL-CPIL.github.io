import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    # Ensure input is numpy array
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    # Calculate return costs for each node j: d[j, 0]
    # Since distance matrix is typically symmetric, d[j, 0] is the same as d[0, j]
    return_costs = distance_matrix[:, 0]  # Shape (N,)
    
    # We need eta[i, j] to depend on d[i, j] and d[j, 0]
    # d[i, j] is distance_matrix[i, j]
    # d[j, 0] is return_costs[j]
    
    # Broadcast return_costs to shape (1, N) to add to each row of distance_matrix
    # distance_matrix is (N, N)
    # return_costs[None, :] is (1, N)
    # total_cost[i, j] = distance_matrix[i, j] + return_costs[j]
    total_cost = distance_matrix + return_costs[None, :]
    
    # Heuristic: Prize / (Total Cost)^2
    # Use epsilon to avoid division by zero
    eps = 1e-9
    prize_matrix = prizes[None, :] + eps
    
    eta = prize_matrix / (total_cost**2 + eps)
    
    # Ensure all entries are positive
    eta = np.maximum(eta, 1e-9)
    
    return eta
