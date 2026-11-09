import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a heuristic matrix for ACO.
    The heuristic favors moves that yield high prize relative to the
    incremental cost imposed by the feasibility check: d[i, j] + d[j, 0].
    """
    # prizes shape: (N,)
    # distance_matrix shape: (N, N)
    
    # Extract distances from every node i to every node j: distance_matrix (N, N)
    # Extract distances from every node j to node 0: distance_matrix[:, 0] (N,)
    # We need to add d[j, 0] to each row i of distance_matrix.
    # d[i, j] + d[j, 0]
    
    d_to_start = distance_matrix[:, 0] # Shape (N,)
    
    # Effective cost for moving i -> j in the context of the feasibility check
    # cost[i, j] = d[i, j] + d[j, 0]
    effective_cost = distance_matrix + d_to_start[None, :]
    
    # Avoid division by zero
    effective_cost = np.maximum(effective_cost, 1e-9)
    
    # Heuristic: Prize of destination j divided by effective cost
    # prizes[None, :] broadcasts to (N, N)
    heuristic = prizes[None, :] / effective_cost
    
    # Ensure non-negative and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
