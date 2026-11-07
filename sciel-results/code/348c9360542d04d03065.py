"""Heuristic matrix for ant colony optimization of orienteering."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Return an edge-attractiveness matrix of the same shape as distance_matrix.
    
    Strategy: For each edge (i, j), compute the marginal gain of adding node j
    given that we've traveled to i. The marginal gain is prize[j] weighted by
    the remaining budget slack after visiting j and returning to the depot.
    This dynamically prioritizes nodes that maximize prize density per unit
    of remaining path budget.
    
    Specifically:
    - slack[i, j] = max_len - d[i, j] - d[j, 0]
    - If slack[i, j] <= 0, the edge is infeasible (can't return), so heuristic is small.
    - Otherwise, heuristic[i, j] = prize[j] * slack[i, j] / (d[i, j] + eps)
      This rewards high-prize nodes that are close to i AND leave enough budget
      to return to the depot.
    
    For node 0 (depot), prize[0] = 0, so edges into node 0 get low heuristic
    unless we're modeling return-to-depot specially. Since ants start at 0 and
    must return to 0, we want to encourage moves that leave budget for return.
    
    We also add a small base term to avoid zero entries.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Compute slack: remaining budget after going i -> j and then j -> 0
    # d[i, j] + d[j, 0] is the cost of visiting j from i and returning to depot
    # The "length so far" when at node i is not known at heuristic computation time,
    # but we can estimate: if we've already spent some budget, the effective max_len
    # is reduced. However, since the heuristic is computed once before the colony runs,
    # we use the full max_len as an upper bound on available budget.
    
    # Cost of edge i->j plus return from j to depot
    cost_to_return = distance_matrix + distance_matrix[:, 0:1]  # n x n: d[i,j] + d[j,0]
    
    # Slack: how much budget remains after taking edge i->j and returning
    slack = max_len - cost_to_return
    slack = np.maximum(slack, 0.0)
    
    # Prize density per unit distance, weighted by slack
    # For each (i, j): prize[j] * slack[i, j] / (d[i, j] + eps)
    prize_col = prizes[None, :]  # 1 x n
    dist_safe = distance_matrix + eps  # n x n
    
    # Marginal gain: high prize, high slack, low distance
    marginal = prize_col * slack / dist_safe
    
    # For edges into node 0, prize[0] = 0, so marginal[*, 0] = 0
    # We want to still allow return to depot, but the heuristic for return edges
    # should be moderate. Since the ant only moves to unvisited nodes and node 0
    # is visited at start, edges into 0 are never actually chosen by ants
    # (they can't revisit 0). So we can set heuristic[*, 0] to a small value.
    
    # Ensure all entries are positive
    heuristic = marginal + eps
    
    # For edges from node 0 (start), we want to encourage good first moves.
    # The above formula already handles this: from node 0, cost_to_return[0, j] = d[0,j] + d[j,0]
    # slack[0, j] = max_len - d[0,j] - d[j,0]
    
    # Clip non-finite values
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, eps)
    
    return heuristic
# EVOLVE-BLOCK-END
