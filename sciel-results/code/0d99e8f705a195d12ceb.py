"""Rewritten program: TSP constructive heuristic with vectorized 2-step lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

def _get_deadline():
    """Return a deadline that is 5 seconds before the 150s limit (safety margin)."""
    return _start_time + 145.0

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: 2-step lookahead with adaptive destination bias.
    Score = dist(current, u) + alpha * min_dist(u, others) + beta * dist(u, destination)
    - min_dist(u, others) encourages picking nodes that are well-connected to the remaining set.
    - beta increases as the tour nears completion to ensure cheap closure.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline with safety margin
    if time.time() > _get_deadline():
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists_from_current)])
    
    # Compute base distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute min distance from each candidate to any other unvisited node
    # This serves as a proxy for the cost of the next step after picking u
    # Vectorized: extract submatrix and find row mins (excluding diagonal)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_matrix, np.inf)
    min_dists = np.min(sub_matrix, axis=1)
    
    # Adaptive parameters
    total_nodes = 50
    progress = 1.0 - (n_unvisited / total_nodes)
    
    # Alpha: weight on lookahead (connectivity to remaining nodes)
    # Constant 0.5 balances immediate path cost with future cluster connectivity
    alpha = 0.5
    
    # Beta: weight on destination proximity
    # Quadratic increase: minimal early on, strong at the end
    beta = 0.5 * progress * progress
    
    # Compute scores
    scores = dists_from_current + alpha * min_dists + beta * dists_to_dest
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
