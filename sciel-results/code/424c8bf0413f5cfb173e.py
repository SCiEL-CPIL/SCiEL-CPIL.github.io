"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 3.0


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP: pick next customer or depot (0) to close route.
    
    Strategy:
      1. Filter unvisited nodes that fit in rest_capacity.
      2. If none fit, return depot.
      3. Score each feasible node using a lookahead metric:
         - immediate distance from current_node
         - plus a fraction of the minimum distance from that node to any other feasible node or depot
         This avoids dead-end picks where a node is close now but far from everything else.
    """
    if _time_exceeded():
        # Safety: return depot to close route quickly
        return 0

    unvisited = list(unvisited_nodes)
    if not unvisited:
        return 0

    # Filter by capacity
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    if not feasible:
        return 0

    # Convert to numpy arrays for vectorized operations
    feasible_arr = np.array(feasible)
    dist_from_current = distance_matrix[current_node, feasible_arr]

    # Lookahead: for each feasible node, compute min distance to any other feasible node or depot
    # We want to avoid nodes that are isolated (far from everything else)
    # Compute distances from each feasible node to all feasible nodes + depot
    # To keep it fast, we do a simple approach: for each feasible node, find min distance to depot or other feasible
    
    n_feasible = len(feasible_arr)
    if n_feasible == 1:
        return int(feasible_arr[0])

    # Build a small distance matrix among feasible nodes + depot
    # indices: feasible_arr plus depot
    indices = np.concatenate([feasible_arr, [depot]])
    sub_dist = distance_matrix[np.ix_(indices, indices)]
    
    # For each feasible node (index 0..n_feasible-1 in sub_dist), find min distance to any other node in sub_dist
    # Exclude self-distance
    lookahead = np.zeros(n_feasible)
    for i in range(n_feasible):
        # min distance from feasible node i to any other node (including depot at index n_feasible)
        # exclude self
        min_dist = np.min(np.delete(sub_dist[i], i))
        lookahead[i] = min_dist

    # Score: immediate distance + alpha * lookahead
    alpha = 0.5
    scores = dist_from_current + alpha * lookahead

    # Pick the node with minimum score
    best_idx = np.argmin(scores)
    return int(feasible_arr[best_idx])
# EVOLVE-BLOCK-END
