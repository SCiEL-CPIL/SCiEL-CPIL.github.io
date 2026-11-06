"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 2.0  # seconds before hard kill


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: pick next customer or depot (0) to close route.
    
    Strategy:
      1. If no unvisited nodes, return depot.
      2. Filter to feasible nodes (demand fits in rest_capacity).
      3. If no feasible nodes, return depot.
      4. Score each feasible node using a combination of:
         - Immediate distance from current_node
         - Lookahead: min distance from candidate to other feasible nodes or depot
         - A penalty for leaving little remaining capacity (encourages grouping)
      5. Return the best-scoring node.
    """
    # Deadline guard
    if time.time() - _start_time > 120.0 - _DEADLINE_MARGIN:
        # Fallback: nearest feasible or depot
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    if len(unvisited_nodes) == 0:
        return depot

    # Filter feasible nodes
    feasible = []
    for n in unvisited_nodes:
        if demands[n] <= rest_capacity:
            feasible.append(n)
    
    if not feasible:
        return depot

    feasible = np.array(feasible)
    
    # If only one feasible node, take it
    if len(feasible) == 1:
        return int(feasible[0])

    # Compute scores
    # Immediate distance from current_node
    imm_dist = distance_matrix[current_node, feasible]
    
    # Lookahead: for each candidate, min distance to any other feasible node or depot
    # We want to avoid "dead-end" picks that force a long return
    # Score = imm_dist + alpha * lookahead
    # lookahead[i] = min( min(dist[feasible[i], feasible[j]] for j != i), dist[feasible[i], depot] )
    
    # Compute distance matrix among feasible nodes and to depot
    # dist_to_others[i] = min distance from feasible[i] to any other feasible node
    # Also consider distance to depot as a fallback
    
    # Efficient computation using numpy
    # dist_feas[i, j] = distance_matrix[feasible[i], feasible[j]]
    dist_feas = distance_matrix[np.ix_(feasible, feasible)]
    
    # For each row i, we want min over j != i
    # Set diagonal to inf, then take min
    np.fill_diagonal(dist_feas, np.inf)
    min_to_other_feas = np.min(dist_feas, axis=1)
    
    # Distance from each feasible node to depot
    dist_to_depot = distance_matrix[feasible, depot]
    
    # Lookahead: min of going to another feasible or going to depot
    lookahead = np.minimum(min_to_other_feas, dist_to_depot)
    
    # Weighted score
    alpha = 0.3
    scores = imm_dist + alpha * lookahead
    
    # Additionally, penalize nodes that consume a large fraction of remaining capacity
    # This encourages packing similar-sized demands together
    cap_penalty = 0.1 * (demands[feasible] / max(rest_capacity, 1))
    scores += cap_penalty
    
    best_idx = np.argmin(scores)
    return int(feasible[best_idx])
# EVOLVE-BLOCK-END
