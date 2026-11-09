import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Calculate centrality: inverse of mean distance to other nodes
    # Sum of distances to all other nodes
    dist_sums = distance_matrix.sum(axis=1) - distance_matrix[np.arange(n), np.arange(n)]
    mean_dists = dist_sums / (n - 1)
    # Inverse centrality (higher is better/more central)
    # Add epsilon to avoid division by zero
    centrality = 1.0 / (mean_dists + 1e-9)
    
    # Base heuristic: Prize / Distance
    # We want a matrix H where H[i, j] is attractiveness of going i -> j
    # Prize is associated with node j
    # Distance is d[i, j]
    
    # prizes shape (n,), distance_matrix shape (n, n)
    # We want (n, n) matrix
    # H[i, j] = (prizes[j] / (distance_matrix[i, j] + eps)) * centrality[j]
    
    # Avoid division by zero for i==j (though ants don't go to visited nodes, 
    # and i==j is visited, but let's be safe)
    eps = 1e-9
    safe_dist = distance_matrix + eps
    safe_dist[np.arange(n), np.arange(n)] = 1.0 # Self distance doesn't matter much, set to 1
    
    prize_col = prizes[:, np.newaxis] # (n, 1)
    centrality_col = centrality[:, np.newaxis] # (n, 1)
    
    # H[i, j] = prize[j] * centrality[j] / dist[i, j]
    H = (prize_col * centrality_col) / safe_dist
    
    # Ensure non-finite values are handled
    H = np.where(np.isfinite(H), H, 1e-9)
    
    # Ensure positive
    H = np.maximum(H, 1e-9)
    
    return H
