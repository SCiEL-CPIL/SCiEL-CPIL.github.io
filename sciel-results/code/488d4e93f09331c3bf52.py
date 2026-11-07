"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization of orienteering.
    
    The heuristic is based on the marginal gain of adding node j to the current best
    feasible tour approximation. We estimate the remaining budget capacity as
    max_len - d[i,j] - d[j,0] and weight prize[j] by this slack, thereby dynamically
    prioritizing nodes that maximize prize density per unit of remaining path budget.
    
    Parameters:
    -----------
    prizes : array-like, shape (n,)
        Prize values for each node. Prize at node 0 is 0.
    distance_matrix : array-like, shape (n, n)
        Distance matrix between nodes.
    max_len : float
        Maximum allowed tour length.
    
    Returns:
    --------
    heuristic : ndarray, shape (n, n)
        Edge-attractiveness matrix (larger is more attractive).
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Create matrices for broadcasting
    d_ij = distance_matrix  # shape (n, n), d_ij[i, j] = d[i, j]
    d_j0 = distance_matrix[:, 0]  # shape (n,), d_j0[j] = d[j, 0]
    
    # Calculate slack: remaining budget if we go i -> j -> 0
    # If slack <= 0, the edge is infeasible even from the start (with 0 length so far)
    slack = max_len - d_ij - d_j0[None, :]
    
    eps = 1e-9
    
    # Feasibility mask: only consider edges that are feasible
    feasible = slack > 0
    
    # Prize value weighting: higher prize = more attractive
    # Use prize directly, scaled by a factor to balance with distance terms
    prize_weight = prizes[None, :]
    
    # Distance penalty: prefer closer nodes (shorter d[i,j])
    # Use exponential decay to strongly favor nearby nodes
    dist_penalty = np.exp(-d_ij / (0.1 * max_len + eps))
    
    # Origin proximity bonus: nodes closer to origin are better since we must return
    origin_proximity = np.exp(-d_j0[None, :] / (0.1 * max_len + eps))
    
    # Slack factor: strongly prefer edges that leave more budget for future moves
    # Use a power to amplify the difference between high and low slack
    slack_factor = np.maximum(slack, 0.0) ** 0.5
    
    # Combine all factors:
    # heuristic = prize * dist_penalty * origin_proximity * slack_factor
    # This balances prize value, closeness to current node, closeness to origin, and remaining budget
    heuristic = prize_weight * dist_penalty * origin_proximity * slack_factor
    
    # Zero out infeasible edges
    heuristic = np.where(feasible, heuristic, 0.0)
    
    # Normalize the heuristic matrix to [0, 1] range to ensure consistent scaling with pheromones
    max_heuristic = np.max(heuristic)
    if max_heuristic > 0:
        heuristic = heuristic / max_heuristic
    
    # Ensure all entries are positive (non-positive become 1e-9)
    # Infeasible edges will have very small values (near 0) and become 1e-9
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
