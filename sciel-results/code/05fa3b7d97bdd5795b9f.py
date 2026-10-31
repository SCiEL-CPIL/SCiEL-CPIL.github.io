"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: evaluator kills after 150s wall-clock.
# We keep a safety margin.
_DEADLINE_SECONDS = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE_SECONDS - _elapsed()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour construction.
    
    Strategy: Use a lookahead heuristic that considers not just the distance to the
    candidate node, but also the expected cost of leaving that node (i.e., the distance
    from the candidate to the destination or to the nearest unvisited node). This is
    a simple 2-opt style lookahead that often outperforms pure nearest-neighbor.
    
    We use a score = dist(current, candidate) + alpha * dist(candidate, destination)
    where alpha is a weight. For uniform random points, a good alpha is around 0.5,
    but we can also consider the nearest neighbor of the candidate among remaining
    unvisited nodes as a proxy for future cost.
    
    To stay fast and within time limits, we use:
      score(i) = dist(current, i) + 0.5 * min(dist(i, dest), dist(i, nearest_unvisited_after_i))
    
    But computing nearest_unvisited_after_i for each candidate is O(n^2) per call,
    which is too slow for 50 cities called 50 times = 50*50*50 = 125k ops, actually fine.
    
    However, simpler: score(i) = dist(current, i) + 0.5 * dist(i, destination)
    This is O(n) per call and often works well.
    
    Let's try a slightly smarter approach: 
      If there are few unvisited nodes left, use exact lookahead.
      Otherwise, use the blended score.
    """
    if len(unvisited_nodes) == 0:
        # Should not happen, but return destination as fallback
        return int(destination_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if _remaining() < 1.0:
        # Fallback to nearest neighbor to be safe
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # Compute distances from current to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    if n_unvisited <= 10:
        # For small sets, do a 1-step lookahead: consider cost to go to candidate, 
        # then to the best next candidate (nearest to candidate among remaining)
        # This is O(n^2) but n is small
        best_score = np.inf
        best_node = unvisited_nodes[0]
        for i, cand in enumerate(unvisited_nodes):
            d_to_cand = d_current[i]
            # Cost from cand to remaining nodes
            d_cand_to_rest = distance_matrix[cand, unvisited_nodes]
            d_cand_to_rest[i] = np.inf  # exclude self
            if len(unvisited_nodes) > 1:
                d_cand_to_next = np.min(d_cand_to_rest)
            else:
                d_cand_to_next = 0.0
            # Also consider going to destination
            d_cand_to_dest = distance_matrix[cand, destination_node]
            # Blend: prefer lower future cost
            future_cost = 0.5 * d_cand_to_next + 0.5 * d_cand_to_dest
            total = d_to_cand + future_cost
            if total < best_score:
                best_score = total
                best_node = cand
        return int(best_node)
    else:
        # For larger sets, use a faster heuristic:
        # score = d(current, cand) + 0.5 * d(cand, destination)
        d_dest = distance_matrix[unvisited_nodes, destination_node]
        scores = d_current + 0.5 * d_dest
        return int(unvisited_nodes[np.argmin(scores)])


# EVOLVE-BLOCK-END
