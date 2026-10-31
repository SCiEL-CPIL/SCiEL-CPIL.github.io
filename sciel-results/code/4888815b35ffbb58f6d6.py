"""Rewritten constructive TSP heuristic: soft-min 2-step lookahead + angle bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 150.0
_START = time.time()
_SOFT_K = 6.0        # temperature for soft-min of next-next distance
_ANGLE_W = 0.35      # weight of the angle (smoothness) penalty
_LOOKAHEAD = 1.0     # weight of the 2-step lookahead term


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Hard deadline guard: fall back to nearest neighbour if we are close to the limit.
    if time.time() > _DEADLINE - 5.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    unvisited = np.asarray(unvisited_nodes, dtype=np.int64)
    n = unvisited.size
    if n == 0:
        return int(current_node)
    if n == 1:
        return int(unvisited[0])

    dm = np.asarray(distance_matrix, dtype=np.float64)

    # First-hop cost from current to each candidate.
    d1 = dm[current_node, unvisited]

    # 2-step lookahead: for each candidate c, the (soft) minimum distance from c
    # to the remaining unvisited cities (excluding c itself).
    # soft-min(x) = exp(-k * min(x)); larger means a smaller min distance (good).
    # We want to prefer candidates whose best onward step is short, so we use
    # the raw min distance as the penalty.
    sub = dm[unvisited, unvisited]            # (n, n)
    np.fill_diagonal(sub, np.inf)
    min_onward = sub.min(axis=1)              # (n,)

    # Angle / smoothness bias: prefer candidates that continue in roughly the
    # same direction as the incoming edge (current -> destination is the previous
    # edge; we want the turn from that edge into the new edge to be small).
    # Reconstruct coordinates from the distance matrix via a stable embedding is
    # expensive; instead use a cheap proxy: the angle between vector
    # (current - destination) and (candidate - current) is small when the tour
    # is smooth. We approximate using distances only via the law of cosines.
    d_prev = float(dm[current_node, destination_node]) if destination_node is not None and destination_node != current_node else 0.0
    d_cand = d1                                   # distance current -> candidate
    d_dest_cand = dm[destination_node, unvisited] # distance destination -> candidate
    if d_prev > 1e-12 and d_cand > 1e-12 and d_dest_cand > 1e-12:
        cosang = (d_prev * d_prev + d_cand * d_cand - d_dest_cand * d_dest_cand) / (2.0 * d_prev * d_cand)
        cosang = np.clip(cosang, -1.0, 1.0)
        # cosang near 1 => small turn (smooth); near -1 => sharp reversal.
        angle_penalty = 1.0 - cosang
    else:
        angle_penalty = np.zeros(n, dtype=np.float64)

    # Combine terms. Normalize each term to [0,1] scale for stable weighting.
    d1_n = d1 / (d1.max() + 1e-12)
    onward_n = min_onward / (min_onward.max() + 1e-12)

    score = (d1_n
             + _LOOKAHEAD * onward_n
             + _ANGLE_W * angle_penalty)

    idx = int(np.argmin(score))
    return int(unvisited[idx])
# EVOLVE-BLOCK-END
