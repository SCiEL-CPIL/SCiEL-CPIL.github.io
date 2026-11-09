"""Rewritten program: savings-based greedy heuristic for CVRP."""
# EVOLVE-BLOCK-START
import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    unvisited = np.asarray(unvisited_nodes, dtype=int)
    if unvisited.size == 0:
        return int(depot)

    # Filter to capacity-feasible customers.
    feas = unvisited[demands[unvisited] <= rest_capacity]
    if feas.size == 0:
        return int(depot)

    # Marginal cost of going current -> c and then (eventually) returning to depot.
    # savings complement: d(cur,c) + d(c,depot) - d(cur,depot)
    d_cur = distance_matrix[current_node, feas]
    d_dep_c = distance_matrix[depot, feas]
    d_cur_dep = distance_matrix[current_node, depot]

    marginal = d_cur + d_dep_c - d_cur_dep

    # Primary: minimize marginal cost. Tie-break: minimize d_cur, then node id for determinism.
    # Combine into a lexicographic sort via a tuple-friendly approach.
    order = np.lexsort((feas, d_cur, marginal))
    best = feas[order[0]]
    return int(best)
# EVOLVE-BLOCK-END
