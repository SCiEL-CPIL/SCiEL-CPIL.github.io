"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Module-level deadline guard: start the clock on first use.
_T0 = None
_DEADLINE = 150.0 - 5.0  # 5s safety margin before the 150s wall-clock kill


def _remaining_time():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return _DEADLINE - (time.time() - _T0)


def _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix):
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = unvisited_nodes.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(unvisited_nodes[0])

    # If time is running out, fall back to cheap nearest-neighbour.
    if _remaining_time() < 0.05:
        return _nearest_node_fallback(current_node, unvisited_nodes, distance_matrix)

    # 1. Immediate edge cost: current -> candidate
    d_cur = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Closure proxy: candidate -> destination (final return leg)
    d_dest = distance_matrix[unvisited_nodes, destination_node]

    # 3. Lookahead: minimum distance from candidate to any other unvisited node
    #    This helps avoid "dead ends" where a node is close to current but far from everything else.
    #    We compute the min over the submatrix of unvisited x unvisited, excluding self (set to inf).
    #    For efficiency with small N (<=50), direct matrix ops are fine.
    #    Note: We want min over j != i for d(i, j) where i is candidate.
    #    Submatrix: distance_matrix[unvisited_nodes, :] is too big if we don't slice.
    #    Let's get the submatrix D_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    #    Then set diagonal to inf and take min along axis 1.
    
    # Optimization: Only do lookahead if n is small enough or if we have time.
    # For n=50, this is a 50x50 slice and min, which is very fast.
    # However, doing this every step (50 times) is 50 * (50*50) ops ~ 125k ops, trivial.
    
    D_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Mask diagonal to prevent selecting self
    np.fill_diagonal(D_sub, np.inf)
    min_next_hop = np.min(D_sub, axis=1)
    
    # If n==2, min_next_hop will be the distance to the other node.
    # If n==1, handled above.

    # Combine scores:
    # Heuristic: d_cur + w1 * min_next_hop + w2 * d_dest
    # Tuning weights: 
    # - d_cur is primary.
    # - min_next_hop penalizes isolated nodes.
    # - d_dest ensures we don't get stuck far from the end.
    
    # Let's try: score = d_cur + 0.5 * min_next_hop + 0.1 * d_dest
    # The previous soft-min was a smoothed version of d_cur, which is redundant since d_cur is already there.
    # Replacing soft-min with min_next_hop adds true 2-step lookahead.
    
    score = d_cur + 0.5 * min_next_hop + 0.1 * d_dest

    idx = int(np.argmin(score))
    return int(unvisited_nodes[idx])
# EVOLVE-BLOCK-END
