"""
Structural change: replace the pre-computed reference-tour follower with a
vectorised 2-lookahead constructive heuristic that evaluates every candidate
node using a small but global cost model.

Strategy:
  1. For each candidate node c in unvisited_nodes:
       cost(c) = dist(current, c)
                + min_{d in unvisited \ {c}} dist(c, d)
                + 0.15 * min_{d in unvisited \ {c}} dist(d, destination)
  2. Pick the candidate with minimum cost.
  3. Fallback to pure nearest neighbour when time budget is exhausted.

This is a different modelling family from the reference-tour approach:
instead of following a pre-optimised tour, it uses an explicit lookahead
score that captures the immediate edge cost, the cost of connecting to the
rest of the unvisited set, and a soft pull toward the destination.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_LIMIT = 140.0  # Safety margin before 150s limit


def _is_deadline_passed():
    return (time.time() - _START_TIME) > _LIMIT


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour using a 2-lookahead heuristic.
    
    Parameters
    ----------
    current_node : int
        The node we are currently at.
    destination_node : int
        The node we must return to at the end.
    unvisited_nodes : np.ndarray
        Array of node indices that have not yet been visited.
    distance_matrix : np.ndarray
        Full distance matrix.
    
    Returns
    -------
    int
        The index of the next node to visit.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline guard: fall back to nearest neighbour
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
    
    # Fast path for small sets
    if n_unvisited <= 2:
        # Just pick the one that minimises total cost including return
        c0 = unvisited_nodes[0]
        c1 = unvisited_nodes[1]
        cost0 = distance_matrix[current_node, c0] + distance_matrix[c0, c1] + distance_matrix[c1, destination_node]
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c0] + distance_matrix[c0, destination_node]
        return int(c0 if cost0 < cost1 else c1)
    
    # Vectorised 2-lookahead scoring
    # dist_current[i] = distance from current_node to unvisited_nodes[i]
    dist_current = distance_matrix[current_node, unvisited_nodes]
    
    # dist_to_dest[i] = distance from unvisited_nodes[i] to destination_node
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For each candidate i, we need:
    #   min_{j != i} dist(unvisited[i], unvisited[j])
    #   min_{j != i} dist(unvisited[j], destination_node)
    
    # Build the sub-distance matrix among unvisited nodes
    sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each row i, find the minimum distance to any other node j != i
    # Set the diagonal to inf, then take row min
    sub_dm_diag_inf = sub_dm.copy()
    np.fill_diagonal(sub_dm_diag_inf, np.inf)
    
    # min_dist_to_other[i] = min distance from unvisited[i] to any other unvisited node
    min_dist_to_other = np.min(sub_dm_diag_inf, axis=1)
    
    # For the destination term: min_{j != i} dist_to_dest[j]
    # This is: if dist_to_dest[i] is the global min, use the 2nd min; else use the global min
    sorted_dest = np.sort(dist_to_dest)
    global_min_dest = sorted_dest[0]
    second_min_dest = sorted_dest[1] if n_unvisited > 1 else 0.0
    
    # For each i, min_dist_to_dest_excl[i] = min over j != i of dist_to_dest[j]
    min_dist_to_dest_excl = np.where(
        dist_to_dest == global_min_dest,
        second_min_dest,
        global_min_dest
    )
    # Handle ties: if multiple nodes have the global min, we need to be more careful
    # But for simplicity, the above works when there's a unique min.
    # Let's do it properly with a second pass for ties
    # Actually, let's just compute it directly for correctness
    # For each i, min over j != i of dist_to_dest[j]
    # This is equivalent to: if dist_to_dest[i] is strictly the minimum, use 2nd min; else use global min
    # But if there are ties at the minimum, we need the 2nd distinct minimum value
    # For n=50 this is fine to compute directly
    
    # More robust: compute min_excl properly
    # For each i, we want min(dist_to_dest[j] for j != i)
    # This is: if dist_to_dest[i] < global_min_dest (impossible) or if there are multiple nodes with global_min_dest
    # Simplest correct approach:
    #   min_excl[i] = min(dist_to_dest with element i removed)
    # We can do: min_excl[i] = min(global_min_dest if dist_to_dest[i] != global_min_dest else second_min_distinct, ...)
    # Let's just compute it with a loop-free approach:
    # Sort dist_to_dest, track indices
    
    # Actually for n<=50, let's just do it vectorised properly:
    # Create a matrix where M[i,j] = dist_to_dest[j] if j != i else inf
    # Then min over j for each i
    # This is O(n^2) but n<=50 so it's 2500 ops, trivial
    
    # Even simpler: 
    # min_excl[i] = min(dist_to_dest) if dist_to_dest[i] > min(dist_to_dest) else min(dist_to_dest excluding i)
    # Let's just use the sorted approach with proper tie handling
    
    # For correctness and simplicity with small n:
    min_dist_to_dest_excl = np.empty(n_unvisited)
    for i in range(n_unvisited):
        if i == 0:
            min_dist_to_dest_excl[i] = np.min(dist_to_dest[1:])
        elif i == n_unvisited - 1:
            min_dist_to_dest_excl[i] = np.min(dist_to_dest[:-1])
        else:
            min_dist_to_dest_excl[i] = np.min(np.concatenate([dist_to_dest[:i], dist_to_dest[i+1:]]))
    
    # Final score
    scores = dist_current + min_dist_to_other + 0.15 * min_dist_to_dest_excl
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
