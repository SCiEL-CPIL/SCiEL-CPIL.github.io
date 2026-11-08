"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using an adaptive multi-factor heuristic:
    - Primary: distance from current node (nearest neighbor)
    - Secondary: distance to destination (increasing weight as tour nears completion)
    - Tertiary: connectivity (min distance to other unvisited nodes, to avoid isolated picks)
    """
    # Time guard: if we're close to deadline, fall back to pure nearest neighbor
    if time.time() - _start_time > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Adaptive weight for destination: small early, larger late
    # When n_unvisited is large (early), alpha ~ 0; when small (late), alpha ~ 0.3
    alpha = 0.3 * (1.0 - n_unvisited / 50.0)
    alpha = max(alpha, 0.02)  # always a tiny bit of destination awareness

    # Beta for connectivity: small constant
    beta = 0.1

    # Distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes]

    # Distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]

    # Connectivity: for each unvisited node, min distance to other unvisited nodes (excluding itself)
    # d_unvisited[i, j] is distance between unvisited[i] and unvisited[j]
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(d_sub, np.inf)
    # If only 2 nodes, each other's min is the other
    d_connect = np.min(d_sub, axis=1)

    # Combined score: lower is better
    scores = d_curr + alpha * d_dest + beta * d_connect

    # Pick the minimum
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
