"""2-step lookahead TSP constructive heuristic with adaptive destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin within 150s wall-clock kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour using 2-step lookahead.
    
    For each candidate j, estimate the cost of:
      - moving current -> j
      - then moving j -> best next node k (or destination if j is last)
    
    An adaptive destination bias increases as fewer nodes remain, pulling
    the tour back toward the start early in the final stretch.
    
    Parameters
    ----------
    current_node : int
    destination_node : int
    unvisited_nodes : np.ndarray
    distance_matrix : np.ndarray
    Returns
    -------
    int : index of the next node to visit
    """
    unvisited = np.asarray(unvisited_nodes)
    n_unvisited = len(unvisited)
    d = distance_matrix
    n_total = d.shape[0]

    if n_unvisited == 0:
        return int(current_node)
    if n_unvisited == 1:
        return int(unvisited[0])

    # Deadline guard: fall back to pure nearest-neighbor
    if time.time() > _DEADLINE - 2.0:
        d_curr = d[current_node, unvisited]
        return int(unvisited[int(np.argmin(d_curr))])

    # Adaptive destination bias: stronger when few nodes remain
    # fraction_remaining = n_unvisited / (n_total - 1)
    # bias goes from ~0.0 (early) to ~0.6 (late)
    frac = n_unvisited / max(n_total - 1, 1)
    dest_bias = 0.6 * (1.0 - frac) ** 2

    # --- 2-step lookahead scoring ---
    # Step 1: distance from current to each candidate
    d_curr = d[current_node, unvisited]  # shape (m,)

    # Step 2: for each candidate j, find the best k != j in unvisited
    # to go to next (or destination if only one left after j)
    m = n_unvisited
    d_unv = d[np.ix_(unvisited, unvisited)]  # m x m
    np.fill_diagonal(d_unv, np.inf)  # exclude self

    # min distance from each j to any other unvisited k
    d_best_next = np.min(d_unv, axis=1)  # shape (m,)

    # If we pick j and it's the last unvisited, we go to destination
    # But we don't know which is last; use the min-next as proxy
    # Blend: if n_unvisited is small, the "next" after j might be destination
    if n_unvisited <= 3:
        # For small sets, directly compute 2-step cost including destination
        d_to_dest = d[unvisited, destination_node]  # shape (m,)
        # 2-step cost for candidate j: d_curr[j] + min(d_unv[j,:], d_to_dest[j])
        # but d_unv[j,:] excludes j itself (inf on diagonal)
        step2 = np.minimum(d_best_next, d_to_dest)
    else:
        step2 = d_best_next

    # Composite: primary is the 2-step path cost, secondary is destination pull
    scores = d_curr + step2 + dest_bias * d[unvisited, destination_node]

    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
