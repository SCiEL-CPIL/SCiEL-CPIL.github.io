"""Capacitated VRP next-node heuristic: savings-based with capacity feasibility."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_MODULE_START = time.time()
TIME_LIMIT = 110.0  # seconds; safety margin below 120s/150s limits


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for capacitated VRP next-node selection.
    
    Uses a savings-based approach: prefer nodes that reduce the total route length
    most when inserted, while respecting capacity constraints.
    """
    # Deadline guard
    elapsed = time.time() - _MODULE_START
    if elapsed > TIME_LIMIT:
        if len(unvisited_nodes) == 0:
            return 0
        feasible = [u for u in unvisited_nodes if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda u: distance_matrix[current_node, u]))

    if len(unvisited_nodes) == 0:
        return 0

    # Filter by capacity feasibility
    feasible_mask = np.array([demands[u] <= rest_capacity for u in unvisited_nodes])
    if not np.any(feasible_mask):
        return 0

    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return 0

    # Compute distances from current node to all feasible nodes
    dists_from_current = distance_matrix[current_node, feasible_nodes]

    # If only one feasible node, pick it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Savings-based scoring:
    # For each candidate node i, the "savings" of visiting it next is:
    #   savings(i) = d(current, i) + d(i, next_after_i) - d(current, next_after_i)
    # But we don't know next_after_i yet. Instead, we approximate:
    #   savings(i) ≈ d(current, i) - d(i, nearest_unvisited_or_depot)
    # This encourages picking nodes that are close to other unvisited nodes.
    
    # A better approach: use the classic savings heuristic idea.
    # The savings of merging two routes is d(depot, i) + d(depot, j) - d(i, j).
    # For next-node selection, we want to minimize the total route length.
    
    # Let's use a composite score:
    # score(i) = alpha * d(current, i) + beta * d(i, nearest_other) - gamma * d(current, nearest_other)
    # where nearest_other is the nearest unvisited node (or depot) to i.
    
    # Actually, let's think about this differently. The total route length is the sum
    # of all edges. When we pick node i next, we add edge (current, i). The remaining
    # route will connect i to the rest. So we want to minimize d(current, i) plus
    # the expected cost of the remaining route from i.
    
    # A good proxy: minimize d(current, i) + d(i, nearest_unvisited_or_depot)
    # This encourages picking nodes that are both close to current and close to
    # other unvisited nodes (cluster formation).
    
    # Compute for each feasible node the minimum distance to any other unvisited node or depot
    scores = np.empty(len(feasible_nodes))
    
    for i, node in enumerate(feasible_nodes):
        d_cur = dists_from_current[i]
        
        # Find min distance from this node to any other unvisited node or depot
        other_unvisited = [u for u in unvisited_nodes if u != node]
        if other_unvisited:
            d_to_others = distance_matrix[node, other_unvisited]
            min_d_others = np.min(d_to_others)
        else:
            min_d_others = 0
        
        d_to_depot = distance_matrix[node, depot]
        min_d_others_depot = min(min_d_others, d_to_depot)
        
        # Composite score: primary distance from current, secondary cluster tightness
        score = d_cur + 0.4 * min_d_others_depot
        scores[i] = score
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
