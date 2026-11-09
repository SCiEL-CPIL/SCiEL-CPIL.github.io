"""TSP constructive heuristic: multi-criteria geometric + lookahead blend."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = None
_DEADLINE = 150.0 - 3.0  # seconds of wall-clock budget we actually use


def _now():
    return time.time()


def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()
    return _DEADLINE - (_now() - _START_TIME)


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next node in a TSP tour construction.

    Uses a combination of:
    1. 1-step nearest neighbor
    2. 2-step lookahead (minimize d(cur,c) + d(c, nearest_neighbor(c)))
    3. Triangle inequality slack (geometric angle proxy)
    4. Destination avoidance bias (keep destination for tour closing)

    The blend is tuned to balance local optimality with global tour shape.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = _now()

    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 0:
        return int(destination_node)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])

    # Time guard: fall back to pure nearest-neighbour if we are close to the limit
    if _time_left() < 0.5:
        d = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[int(np.argmin(d))])

    if n_unvisited == 2:
        a, b = unvisited_nodes[0], unvisited_nodes[1]
        cost_a = distance_matrix[current_node, a] + distance_matrix[a, b]
        cost_b = distance_matrix[current_node, b] + distance_matrix[b, a]
        return int(a) if cost_a <= cost_b else int(b)

    # --- Compute various score components ---
    
    # 1. Direct distance from current to each candidate
    d_cur_cand = distance_matrix[current_node, unvisited_nodes]  # (n,)
    
    # 2. 2-step lookahead: for each candidate c, find its nearest neighbor among unvisited
    #    and compute d(cur,c) + d(c, nn(c))
    #    We approximate this by looking at the minimum distance from each candidate to any other unvisited node
    d_cand_cand = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]  # (n, n)
    # Set diagonal to inf so we don't pick self
    np.fill_diagonal(d_cand_cand, np.inf)
    nearest_dist = np.min(d_cand_cand, axis=1)  # (n,)
    lookahead_score = d_cur_cand + nearest_dist  # (n,)
    
    # 3. Triangle inequality slack: d(dest,cur) + d(cur,c) - d(dest,c)
    d_dc = distance_matrix[destination_node, current_node]
    d_dest_cand = distance_matrix[destination_node, unvisited_nodes]  # (n,)
    slack = d_dc + d_cur_cand - d_dest_cand
    np.maximum(slack, 0.0, out=slack)
    
    # Normalize scores to [0, 1] range for blending
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    norm_nn = normalize(d_cur_cand)
    norm_lookahead = normalize(lookahead_score)
    norm_slack = normalize(slack)
    
    # Blend: weights tuned for Euclidean TSP
    # Heavier weight on lookahead (captures 2-step structure), moderate on slack (geometry), light on pure NN
    w_nn = 0.2
    w_lookahead = 0.5
    w_slack = 0.3
    
    scores = w_nn * norm_nn + w_lookahead * norm_lookahead + w_slack * norm_slack
    
    # Destination avoidance: if destination is in unvisited and we're not near the end,
    # penalize it to keep it for closing the tour
    dest_mask = (unvisited_nodes == destination_node)
    if np.any(dest_mask) and n_unvisited > 4:
        scores[dest_mask] += 0.15
    
    best_idx = int(np.argmin(scores))
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
