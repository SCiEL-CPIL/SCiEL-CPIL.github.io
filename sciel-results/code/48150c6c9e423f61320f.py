"""TSP constructive heuristic combining geometric alignment with adaptive connectivity."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0  # Safety margin before 120s limit

def _is_deadline_passed():
    return (time.time() - _start_time) > _LIMIT

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a hybrid geometric alignment and connectivity heuristic.
    
    This approach combines two structural ideas:
    1. Geometric Alignment (from Inspiration 1): Uses the triangle inequality proxy
       (d(cur,c) + d(c,dest) - d(cur,dest)) to measure how well a candidate lies
       on the path toward the destination. This encourages smooth, non-crossing paths.
    2. Adaptive Connectivity (from Current Program): Uses the average distance from
       a candidate to all other unvisited nodes as a "centrality" penalty. This
       prevents the tour from expanding too far or creating isolated clusters.
    
    The final score is a weighted sum of normalized alignment and normalized connectivity,
    with weights that adapt based on tour progress. Early in the tour, we favor
    connectivity to build a compact core. Late in the tour, we favor alignment to
    ensure we can close the loop efficiently.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        # Exact 2-node optimization: pick the one that minimizes total remaining cost
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c1, destination_node]
        return c1 if cost1 < cost2 else c2

    if _is_deadline_passed():
        # Fallback to pure nearest neighbor
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(dists_from_current)
        return int(unvisited_nodes[best_idx])

    # Precompute distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_cur_dest = distance_matrix[current_node, destination_node]
    
    # --- Geometric Alignment Score ---
    # Alignment proxy: (d(cur, c) + d(c, dest) - d(cur, dest))
    # Lower means c is more "aligned" with the direction to destination.
    # This is 0 if c lies on the straight line between cur and dest.
    alignment_scores = dists_from_current + dists_to_dest - dist_cur_dest
    
    # Normalize alignment scores to [0, 1]
    min_align = np.min(alignment_scores)
    max_align = np.max(alignment_scores)
    align_range = max_align - min_align
    if align_range > 1e-10:
        normalized_align = (alignment_scores - min_align) / align_range
    else:
        normalized_align = np.zeros_like(alignment_scores)
    
    # --- Connectivity Score ---
    # Average distance from each candidate to all other unvisited nodes.
    # This acts as a "centrality" penalty, preferring nodes that are close to the rest of the cluster.
    if n_unvisited > 1:
        submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Sum of distances from each candidate to all other unvisited nodes
        # Exclude self (diagonal is 0 anyway)
        connectivity_costs = np.sum(submatrix, axis=1)
        avg_connectivity = connectivity_costs / max(1, n_unvisited - 1)
        
        # Normalize connectivity to [0, 1]
        min_conn = np.min(avg_connectivity)
        max_conn = np.max(avg_connectivity)
        conn_range = max_conn - min_conn
        if conn_range > 1e-10:
            normalized_conn = (avg_connectivity - min_conn) / conn_range
        else:
            normalized_conn = np.zeros_like(avg_connectivity)
    else:
        normalized_conn = np.zeros_like(dists_from_current)
    
    # --- Adaptive Weights ---
    # Estimate total nodes N
    N = distance_matrix.shape[0]
    progress = 1.0 - (n_unvisited / N)
    
    # Early in the tour (low progress): Favor connectivity to build a compact core.
    # Late in the tour (high progress): Favor alignment to close the loop efficiently.
    #
    # We want:
    # - w_conn: High early, low late.
    # - w_align: Low early, high late.
    #
    # Let's use a smooth transition.
    # w_conn = 0.6 - 0.4 * progress
    # w_align = 0.4 + 0.4 * progress
    #
    # This ensures w_conn + w_align = 1.0 always.
    
    w_conn = 0.6 - 0.4 * progress
    w_align = 0.4 + 0.4 * progress
    
    # Ensure weights are non-negative
    w_conn = max(0.0, w_conn)
    w_align = max(0.0, w_align)
    
    # Composite score
    scores = w_conn * normalized_conn + w_align * normalized_align
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
