import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic matrix for ant colony orienteering.
    
    Slack-weighted prize density: prioritize edges that lead to high-prize
    nodes while preserving enough budget to return. The marginal gain of
    adding node j from i is weighted by the remaining path budget after
    visiting j and returning to origin.
    
    eta[i,j] = prize[j] * (max_len - d[i,j] - d[j,0]) / (d[i,j] + d[j,0])
    
    This dynamically prioritizes nodes that maximize prize density per unit
    of remaining path budget, directly aligning with the feasibility constraint.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    eps = 1e-9
    
    # detour[i, j] = distance_matrix[i, j] + distance_matrix[j, 0]
    # distance_matrix[j, 0] is shape (N,), broadcast along rows
    detour = distance_matrix + distance_matrix[:, 0][None, :]
    
    # slack[i, j] = max_len - detour[i, j]
    # Only positive slack values are feasible; clip to ensure non-negativity
    slack = np.maximum(max_len - detour, 0.0)
    
    # prize_term: (N,) broadcast to (N, N) along axis 0
    prize_term = prizes[None, :]
    
    # Combine: prize * slack / detour
    # This gives high values for high-prize nodes with large remaining budget
    heuristic = prize_term * slack / (detour + eps)
    
    # Ensure validity: non-positive or non-finite values become 1e-9
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
