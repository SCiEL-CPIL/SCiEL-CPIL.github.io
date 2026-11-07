import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    start_time = time.time()
    timeout = 300.0
    
    try:
        # Ensure inputs are numpy arrays for efficient computation
        prizes = np.asarray(prizes, dtype=float)
        distance_matrix = np.asarray(distance_matrix, dtype=float)
        
        # Distance from each node to the depot (node 0)
        # distance_matrix[:, 0] gives a vector of distances d[i, 0] for all i
        # Since the matrix is symmetric, d[i, 0] == d[0, i]
        dist_to_depot = distance_matrix[:, 0]
        
        # Construct a cost matrix where cost[i, j] represents the "expense"
        # of moving from i to j and then returning to the depot.
        # This aligns with the constraint: current_len + d[i, j] + d[j, 0] <= max_len
        # By using d[i, j] + d[j, 0] in the denominator, we penalize moves that
        # consume a large portion of the remaining path budget, especially
        # for nodes that are far from the depot.
        
        # Broadcast dist_to_depot to match the shape of distance_matrix
        # dist_to_depot[None, :] has shape (1, N), distance_matrix has (N, N)
        # Resulting cost_matrix[i, j] = distance_matrix[i, j] + dist_to_depot[j]
        cost_matrix = distance_matrix + dist_to_depot[None, :]
        
        # Calculate heuristic: Prize of destination node j divided by the cost
        # prizes[None, :] has shape (1, N), broadcasting to (N, N)
        eps = 1e-9
        eta = prizes[None, :] / (cost_matrix + eps)
        
        # Sanitize: Ensure all values are positive and finite
        # The task specifies: "Heuristic entries that are not positive become 1e-9"
        eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
        
        return eta

    except Exception as e:
        # If anything goes wrong, return a safe default matrix
        # This prevents the whole evaluation from crashing due to a heuristic error
        n = len(prizes) if hasattr(prizes, '__len__') else 50
        return np.full((n, n), 1e-9)
