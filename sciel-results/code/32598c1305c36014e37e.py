"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Strategy: Lookahead-2 with Connectivity Penalty
#
# The current approach (Prim-like / MST-growth) uses a 1-step greedy choice with a
# structural penalty based on min-distance to other unvisited nodes. This is still
# fundamentally 1-step greedy.
#
# This new strategy uses a 2-step lookahead: for each candidate c, we estimate the
# cost of taking the edge (current -> c) PLUS the best possible next edge (c -> c')
# where c' is the closest unvisited node to c (excluding c itself). This captures
# the "cost of commitment" - choosing a node that leads to a bad next step is penalized.
#
# Additionally, we add a global connectivity penalty: the average distance from c to
# all other unvisited nodes. This discourages choosing nodes that are on the "fringe"
# of the point cloud, which would force long edges later in the tour.
#
# Score(c) = dist(current, c) + w1 * min_{u != c} dist(c, u) + w2 * mean_{u != c} dist(c, u)
#
# This is materially different from:
# 1. Pure Nearest Neighbor (no lookahead, no global term)
# 2. Current Prim-like (uses min-distance as penalty but no 2-step lookahead in the primary cost)
# 3. Bottleneck (uses max instead of mean, no lookahead)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a 2-step lookahead with global connectivity penalty.
    
    For each candidate c:
    - Primary cost: dist(current, c)
    - Lookahead cost: min distance from c to any other unvisited node (best next step)
    - Connectivity penalty: mean distance from c to all other unvisited nodes
    
    This balances immediate cost, future flexibility, and global integration.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_remaining = len(unvisited_nodes)
    
    # Deadline guard: track elapsed time for safety
    t_start = time.time()
    DEADLINE = 145.0  # 5s safety margin before 150s kill
    
    # 1. Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Sub-matrix of distances between unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For min and mean, exclude self (diagonal)
    sub_matrix_no_diag = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_no_diag, np.inf)
    
    # 3. Lookahead: min distance from each candidate to any other unvisited node
    min_dist_to_other = np.min(sub_matrix_no_diag, axis=1)
    
    # 4. Connectivity penalty: mean distance from each candidate to all other unvisited nodes
    # Replace inf (self) with 0 for mean calculation, or use sum/(n-1)
    # Sum over axis=1, then divide by (n_remaining - 1)
    sum_dist_to_others = np.sum(sub_matrix_no_diag, axis=1)
    mean_dist_to_other = sum_dist_to_others / max(n_remaining - 1, 1)
    
    # 5. Combine scores
    # Weights: 
    #   w1 = 1.0 for lookahead (equally important as immediate step)
    #   w2 = 0.3 for connectivity penalty (moderate global awareness)
    # These are chosen to balance the three terms without being overly aggressive.
    
    w1 = 1.0
    w2 = 0.3
    
    scores = dists_from_current + w1 * min_dist_to_other + w2 * mean_dist_to_other
    
    # Check deadline before returning
    if time.time() > DEADLINE:
        # Safety: return nearest neighbor as fallback
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
