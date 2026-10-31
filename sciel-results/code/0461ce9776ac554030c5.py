"""Structural change: MST-aware greedy with soft-min smoothing and adaptive destination pull."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin within 150s wall-clock kill


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city using an MST-aware greedy heuristic with soft-min smoothing.
    
    Key idea: avoid creating long 'bridge' edges by preferring nodes that keep
    the remaining graph well-connected. We use a soft-min over distances to
    approximate the MST structure and penalize nodes that would create
    disconnected components.
    
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

    # Precompute distances
    d_curr = d[current_node, unvisited]  # shape (m,)
    d_dest = d[unvisited, destination_node]  # shape (m,)
    d_direct = d[current_node, destination_node]

    m = n_unvisited
    # d[j, k] for all j, k in unvisited
    d_unvisited = d[np.ix_(unvisited, unvisited)]  # shape (m, m)

    # MST-aware score: for each candidate j, estimate the cost of adding j to the tour
    # while keeping the remaining graph well-connected.
    
    # For each j, compute the minimum distance from j to any other unvisited node (excluding self)
    # This approximates how "connected" j is to the rest of the graph
    d_unvisited_safe = d_unvisited.copy()
    np.fill_diagonal(d_unvisited_safe, np.inf)
    min_conn_from_j = np.min(d_unvisited_safe, axis=1)  # shape (m,)
    
    # Also compute the second-minimum to get a sense of redundancy in connections
    # Sort each row and take the second smallest
    sorted_d = np.sort(d_unvisited_safe, axis=1)
    second_min_conn = sorted_d[:, 1] if m > 2 else np.full(m, np.inf)
    
    # Soft-min smoothing: use a soft minimum to get a smoother estimate of connection quality
    # Lower soft-min means better connected
    tau = 0.5  # temperature for soft-min
    # Soft-min of d_unvisited_safe[j, :] over k != j
    # Use log-sum-exp trick for numerical stability
    log_softmin = np.log(np.sum(np.exp(-d_unvisited_safe / tau), axis=1) * tau)
    softmin_conn = -log_softmin  # shape (m,)
    
    # MST-aware cost: prefer nodes that are close to current AND well-connected to the rest
    # The idea: if j is isolated (high min_conn_from_j), adding it now might create a bridge
    # later when we have to connect it back, which is expensive.
    
    # Base cost: distance from current to j
    base_cost = d_curr.copy()
    
    # Connection penalty: penalize nodes that are poorly connected to the remaining set
    # Normalize connection quality
    conn_min, conn_max = np.min(min_conn_from_j), np.max(min_conn_from_j)
    if conn_max > conn_min:
        conn_norm = (min_conn_from_j - conn_min) / (conn_max - conn_min)
    else:
        conn_norm = np.zeros_like(min_conn_from_j)
    
    # Soft-min connection penalty (smoother)
    sm_min, sm_max = np.min(softmin_conn), np.max(softmin_conn)
    if sm_max > sm_min:
        sm_norm = (softmin_conn - sm_min) / (sm_max - sm_min)
    else:
        sm_norm = np.zeros_like(softmin_conn)
    
    # Destination proximity
    d_dest_max = np.max(d_dest) + 1e-9
    dest_score = d_dest / d_dest_max
    
    # Adaptive weights based on how many nodes remain
    frac = n_unvisited / max(n_total - 1, 1)
    
    # Early in the tour: weight connection quality heavily to avoid bridges
    # Late in the tour: weight destination proximity heavily
    w_base = 0.4 * frac ** 0.5
    w_conn = 0.3 * frac ** 0.7
    w_sm = 0.1 * frac ** 0.7
    w_dest = 1.0 - w_base - w_conn - w_sm
    if w_dest < 0:
        w_dest = 0.0
        total = w_base + w_conn + w_sm
        if total > 0:
            w_base /= total
            w_conn /= total
            w_sm /= total
        else:
            w_base, w_conn, w_sm, w_dest = 0.0, 0.0, 0.0, 1.0
    
    # Normalize base cost
    base_min, base_max = np.min(base_cost), np.max(base_cost)
    if base_max > base_min:
        base_norm = (base_cost - base_min) / (base_max - base_min)
    else:
        base_norm = np.zeros_like(base_cost)
    
    # Composite score: minimize (lower is better)
    scores = (w_base * base_norm + 
              w_conn * conn_norm + 
              w_sm * sm_norm + 
              w_dest * dest_score)
    
    best_idx = int(np.argmin(scores))
    return int(unvisited[best_idx])
# EVOLVE-BLOCK-END
