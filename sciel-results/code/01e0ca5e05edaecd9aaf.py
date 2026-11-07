import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ant colony optimization.
    
    Args:
        prizes: 1D array of node prizes.
        distance_matrix: 2D array of pairwise distances.
        max_len: Maximum tour length.
        
    Returns:
        2D array of heuristic values (edge attractiveness).
    """
    n = len(prizes)
    
    # 1. Compute a 'hub score' for each node.
    # This rewards nodes that are close to other high-value nodes.
    # We use a threshold to define 'neighbors'.
    # With 50 points in a unit square, avg dist is ~0.14. 
    # Let's use a threshold of 0.25 to capture local density.
    threshold = 0.25
    mask = (distance_matrix < threshold) & (distance_matrix > 0)
    
    # hub_score[j] = sum of prizes of nodes i that are close to j
    # Using matrix multiplication: mask.T @ prizes
    # mask[i,j] is 1 if i is close to j.
    # We want for each j, sum of prizes[i] where mask[i,j] is 1.
    # This is mask.T @ prizes.
    hub_scores = mask.T @ prizes
    
    # Combine direct prize with hub potential
    # Weight the hub score to not overpower the direct prize
    effective_value = prizes + 0.5 * hub_scores
    
    # 2. Compute the heuristic matrix
    # Heuristic eta[i, j] ~ effective_value[j] / cost(i, j)
    # Cost is approximated by d[i, j] + d[j, 0]
    # d[j, 0] is the return cost.
    
    # Create a matrix of return costs for each destination j
    # return_costs[j] = d[j, 0]
    return_costs = distance_matrix[:, 0]
    
    # We need eta[i, j] for all i, j.
    # Denominator[i, j] = distance_matrix[i, j] + return_costs[j]
    
    # To avoid division by zero or very small numbers, add epsilon
    eps = 1e-9
    
    denom = distance_matrix + return_costs[None, :] + eps
    numer = effective_value[None, :] + eps
    
    # Calculate heuristic
    H = numer / denom
    
    # Ensure positive values
    H = np.maximum(H, eps)
    
    # Ensure finite values
    H = np.where(np.isfinite(H), H, eps)
    
    return H
