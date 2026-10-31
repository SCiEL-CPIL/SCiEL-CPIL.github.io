# EVOLVE-BLOCK-START
import numpy as np
import time

# Wall-clock deadline guard with safety margin (evaluator kills at 150s; we target 120s).
_DEADLINE = 115.0  # seconds from process start; safety margin before hard limit
_START = time.time()


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard: if we're close to the limit, use the cheapest safe choice.
    elapsed = time.time() - _START
    if elapsed > _DEADLINE:
        # Pure nearest neighbor from current_node as a safe fallback.
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    n = len(unvisited_nodes)
    if n == 0:
        # No unvisited nodes; return current as a safe default (shouldn't happen in valid tours).
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # --- Potential field model ---
    # The next node minimizes: attractive term (distance to current) + repulsive
    # potential (inverse-distance sum over other unvisited nodes). The repulsive
    # term pushes the tour toward spatially dispersed nodes, preventing early
    # saturation of local clusters and improving global coverage.

    # Attractive: distance from current node to each candidate.
    edge = distance_matrix[current_node, unvisited_nodes]  # (n,)

    # Repulsive: for each candidate i, sum of 1/dist(i, j) over all other
    # unvisited nodes j. This is computed via the full sub-matrix among
    # unvisited nodes.
    sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
    # Set diagonal to 1 to avoid division by zero; we'll zero it out after.
    diag_idx = np.diag_indices(n)
    sub[diag_idx] = 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = 1.0 / sub
    inv[diag_idx] = 0.0  # remove self-interaction
    repulsive = inv.sum(axis=1)  # (n,)

    # Scale the repulsive term to be comparable to edge distances.
    # Use median edge as a reference scale for normalization.
    edge_med = np.median(edge) if n > 0 else 1.0
    if edge_med < 1e-12:
        edge_med = 1.0

    # Adaptive repulsion strength: stronger when fewer nodes remain (to spread
    # out the remaining choices more aggressively), weaker early on.
    # For 50 cities, n starts at 50 and goes to 2.
    # gamma scales the repulsive potential.
    gamma = 0.15 * (50.0 / max(n, 1.0))  # stronger repulsion as n shrinks

    # Normalize repulsive by its median to keep scale stable.
    rep_med = np.median(repulsive) if n > 0 else 1.0
    if rep_med < 1e-12:
        rep_med = 1.0

    score = edge + gamma * repulsive * (edge_med / rep_med)

    idx = int(np.argmin(score))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
