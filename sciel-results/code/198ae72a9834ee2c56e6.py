"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    # Deadline guard: if we're running low on time, fall back to nearest neighbor
    elapsed = time.time() - _start_time
    if elapsed > 110.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Total number of cities is inferred from distance_matrix size
    n_cities = distance_matrix.shape[0]

    # Distances from current node to all unvisited candidates
    d_current_cand = distance_matrix[current_node, unvisited_nodes]

    # Distances from each candidate to the destination
    d_cand_dest = distance_matrix[destination_node, unvisited_nodes]

    # Fraction of tour remaining (1.0 at start, ~0 near end)
    frac_remaining = n_unvisited / n_cities

    # Stranding penalty: for each candidate, what's the max distance from current
    # to the OTHER unvisited nodes (excluding the candidate)?
    # This is expensive if done naively, but n_unvisited <= 50 so it's fine.
    # We compute: for candidate at index i in unvisited_nodes,
    #   remaining = unvisited_nodes with that element removed
    #   max_d = max(distance_matrix[current_node, remaining]) if remaining else 0

    # Vectorized approach: create a matrix of distances from current to all unvisited
    # For each candidate i, we want max of d_current_cand excluding index i.
    # We can compute this by: for each i, max of d_current_cand with d_current_cand[i] set to -inf.
    # Since n_unvisited is small (<=50), we can do it with a loop or vectorized.

    # Efficient: the max excluding index i is either the global max (if argmax != i)
    # or the second-max (if argmax == i).
    d_sorted = np.sort(d_current_cand)
    global_max = d_sorted[-1]
    second_max = d_sorted[-2] if n_unvisited >= 2 else 0.0
    argmax_idx = np.argmax(d_current_cand)

    # For each candidate index i, max_excluding_i = global_max if i != argmax_idx else second_max
    max_excluding = np.where(np.arange(n_unvisited) == argmax_idx, second_max, global_max)

    # Weights: stranding penalty should be moderate; destination penalty grows as we near end
    w_strand = 0.5
    w_dest = 1.0 * (1.0 - frac_remaining)  # increases from 0 to ~1 as tour nears completion

    # Total score for each candidate
    scores = d_current_cand + w_strand * max_excluding + w_dest * d_cand_dest

    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
