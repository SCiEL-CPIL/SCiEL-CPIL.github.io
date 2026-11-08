"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in a TSP tour using a lookahead heuristic that
    balances immediate distance, distance to destination (for closing),
    and a 2-step lookahead to avoid poor local choices.
    """
    # Deadline guard: if we're running out of time, fall back to nearest neighbor
    if time.time() - _start_time > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    unvisited = unvisited_nodes
    n_unvisited = len(unvisited)

    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited[0])

    # Distances from current node to all unvisited nodes
    d_from_current = distance_matrix[current_node, unvisited]

    # Distances from all unvisited nodes to destination (for closing cost consideration)
    d_to_dest = distance_matrix[unvisited, destination_node]

    # If only 2 nodes left, pick the one that minimizes the total closing cost:
    # cost = d(current->i) + d(i->j) + d(j->dest)
    if n_unvisited == 2:
        i, j = unvisited
        cost_ij = distance_matrix[current_node, i] + distance_matrix[i, j] + distance_matrix[j, destination_node]
        cost_ji = distance_matrix[current_node, j] + distance_matrix[j, i] + distance_matrix[i, destination_node]
        return int(i if cost_ij <= cost_ji else j)

    # For 3+ nodes, use a scoring heuristic:
    # score = d_from_current[i] + alpha * d_to_dest[i] + beta * lookahead_penalty
    # where lookahead_penalty considers the best 2nd-step option from each candidate

    # Compute for each candidate i: the minimum distance from i to any other unvisited node
    # (excluding i itself) — this approximates the cost of continuing the tour
    # We use a vectorized approach: for each unvisited node, find min distance to other unvisited
    # This is an n_unvisited x n_unvisited sub-matrix operation

    # Get the sub-matrix of distances among unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited, unvisited)]
    
    # For each row (candidate), the min distance to any other unvisited node
    # We need to exclude the diagonal (self-distance)
    np.fill_diagonal(sub_matrix, np.inf)
    min_next_from_candidate = np.min(sub_matrix, axis=1)
    np.fill_diagonal(sub_matrix, 0)  # restore for potential reuse (not needed here but safe)

    # Also consider: after going to candidate i, the best 2nd step is min distance from i to others
    # But we also want to eventually close to destination, so weight d_to_dest

    # Heuristic: 
    # primary: distance from current to candidate
    # secondary: distance from candidate to destination (encourages moving toward destination)
    # tertiary: penalty if candidate has no good next options (high min_next_from_candidate)
    
    # Use a weighted sum with tunable parameters
    # For 50-city uniform instances, empirical tuning suggests:
    # - Strong weight on immediate distance
    # - Moderate weight on destination distance (especially as we get closer to end)
    # - Small weight on lookahead penalty
    
    # Adaptive alpha: as fewer nodes remain, weight destination distance more heavily
    # When n_unvisited is large, focus on immediate distance
    # When n_unvisited is small, focus on closing
    
    if n_unvisited <= 5:
        alpha = 0.5
        beta = 0.3
    elif n_unvisited <= 10:
        alpha = 0.3
        beta = 0.2
    elif n_unvisited <= 20:
        alpha = 0.2
        beta = 0.1
    else:
        alpha = 0.1
        beta = 0.05

    # Normalize components to comparable scales
    d_from_current_norm = d_from_current / (np.max(d_from_current) + 1e-9)
    d_to_dest_norm = d_to_dest / (np.max(d_to_dest) + 1e-9)
    min_next_norm = min_next_from_candidate / (np.max(min_next_from_candidate) + 1e-9)

    scores = d_from_current_norm + alpha * d_to_dest_norm + beta * min_next_norm

    best_idx = np.argmin(scores)
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
