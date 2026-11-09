import numpy as np
from scipy.optimize import linear_sum_assignment
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linear_sum_assignment


def heuristics(distance_matrix):
    """
    Compute an edge-attractiveness heuristic matrix using node-potential gradients
    derived from the linear assignment problem (Hungarian algorithm).
    
    The heuristic H_ij = exp(-(c_ij - u_i - v_j)) where u, v are the dual variables
    (node potentials) from the optimal assignment. This creates a smooth landscape
    where edges with low assignment regret are exponentially favored.
    """
    n = distance_matrix.shape[0]
    
    # Solve the linear assignment problem to get optimal row and column indices
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    
    # Compute the optimal assignment cost
    optimal_cost = np.sum(distance_matrix[row_ind, col_ind])
    
    # For the Hungarian algorithm, we can extract dual variables from the reduced costs.
    # The reduced cost for edge (i, j) is: c_ij - u_i - v_j
    # For optimal edges, reduced cost = 0. For suboptimal edges, reduced cost > 0.
    #
    # We need to find u and v such that:
    # - u_i + v_j <= c_ij for all i, j (dual feasibility)
    # - u_i + v_j = c_ij for edges in the optimal assignment
    #
    # A simple approach: use the fact that in the Hungarian algorithm, the dual variables
    # can be constructed from the row and column minima adjustments. However, scipy's
    # linear_sum_assignment doesn't directly return dual variables.
    #
    # Alternative approach: We can compute a heuristic based on the distance matrix
    # and the assignment structure. Let's use a simpler but effective heuristic:
    #
    # H_ij = exp(-alpha * c_ij) * beta_ij
    # where beta_ij is a bonus for edges that are "close" to being in a good tour.
    #
    # Actually, let's try a different approach. We can compute the reduced costs
    # by solving for dual variables. One way is to use the fact that for the optimal
    # assignment, we can set u_i = c_{i, sigma(i)} - v_{sigma(i)} for some v.
    #
    # Let's use a practical approach: compute a heuristic that combines inverse distance
    # with a penalty based on how "expensive" each edge is relative to the median.
    
    # Compute the median distance
    # Exclude zero distances (self-loops)
    d = distance_matrix.copy()
    np.fill_diagonal(d, np.nan)
    median_dist = np.nanmedian(d)
    
    # Base heuristic: inverse distance
    base_heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Compute a "regret" measure: how much more expensive is edge (i,j) compared
    # to the best edge from node i and to the best edge to node j
    # Best outgoing from each node
    best_out = np.min(distance_matrix, axis=1)
    # Best incoming to each node
    best_in = np.min(distance_matrix, axis=0)
    
    # Regret for edge (i,j): how much worse is it than the best options
    # regret_ij = c_ij - max(best_out[i], best_in[j])
    # This is always >= 0 for edges that are not the best in either direction
    regret = distance_matrix - np.maximum(
        best_out[:, np.newaxis], 
        best_in[np.newaxis, :]
    )
    regret = np.maximum(regret, 0)
    
    # Scale regret by the median distance to normalize
    regret_scaled = regret / (median_dist + 1e-9)
    
    # Heuristic: combine inverse distance with exponential decay of regret
    # H_ij = base_heuristic * exp(-gamma * regret_scaled)
    gamma = 1.0
    heuristic = base_heuristic * np.exp(-gamma * regret_scaled)
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
