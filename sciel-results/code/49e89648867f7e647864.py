"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache to track the previous node for directional momentum
_prev_node = None
_instance_key = None
_tour_length = 0  # number of nodes visited so far in current tour
_total_nodes = 50  # default assumption; will be inferred from distance_matrix shape


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _prev_node, _instance_key, _tour_length, _total_nodes

    if len(unvisited_nodes) == 0:
        return -1
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Time guard: fall back to pure NN if running long
    t0 = time.time()
    if t0 > 145.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

    # Detect new instance (different matrix shape or size)
    n = distance_matrix.shape[0]
    key = (n, current_node, destination_node)
    if key != _instance_key:
        _instance_key = key
        _tour_length = 0
        _prev_node = None
        _total_nodes = n

    _tour_length += 1

    # Compute base distances
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[unvisited_nodes, destination_node]

    # Phase-aware alpha: early tour → more NN (alpha high), late tour → more destination-aware
    # This avoids long closing edges
    progress = _tour_length / max(_total_nodes - 1, 1)
    # alpha goes from 0.85 early to 0.5 late
    alpha = 0.85 - 0.35 * progress

    scores = alpha * dists_from_current + (1 - alpha) * dists_to_dest

    # Directional momentum penalty: penalize sharp turns
    if _prev_node is not None:
        # Vector from prev to current
        v_in = distance_matrix[_prev_node, current_node]  # scalar dist, but we need direction
        # We don't have coordinates, only distance matrix.
        # Use a proxy: the "turn cost" is approximated by how much the candidate
        # diverges from the straight line prev→current→candidate.
        # In a Euclidean metric, |prev-cand| ≈ |prev-cur| + |cur-cand| when collinear (no turn).
        # A sharp turn means |prev-cand| >> |prev-cur| + |cur-cand| is false;
        # actually for a straight line: d(prev,cand) = d(prev,cur) + d(cur,cand).
        # For a turn: d(prev,cand) < d(prev,cur) + d(cur,cand) by triangle inequality.
        # The "excess" is d(prev,cur) + d(cur,cand) - d(prev,cand), which is 0 for straight, >0 for turn.
        # So we penalize larger excess → prefer straighter paths.

        d_prev_cur = distance_matrix[_prev_node, current_node]
        d_prev_cand = distance_matrix[_prev_node, unvisited_nodes]
        # excess = d_prev_cur + dists_from_current - d_prev_cand  (>= 0 by triangle inequality)
        excess = d_prev_cur + dists_from_current - d_prev_cand
        # Normalize excess by d_prev_cur to get a relative turn measure
        # Small excess relative to d_prev_cur → straighter → lower penalty
        turn_penalty = excess / max(d_prev_cur, 1e-9)
        # Blend turn penalty into scores
        scores = scores + 0.15 * turn_penalty

    # 1-step lookahead: for top-3 candidates, estimate cost of going candidate → best-next-from-candidate
    # This reduces short-sightedness without full lookahead
    n_cand = len(unvisited_nodes)
    if n_cand <= 30 and t0 < 140.0:
        # Get top-3 by current score
        top_k = min(3, n_cand)
        top_indices = np.argsort(scores)[:top_k]
        for idx in top_indices:
            cand = unvisited_nodes[idx]
            # Best next from candidate (excluding current and already-visited)
            # We can approximate: best unvisited from cand, but we don't know visited set fully.
            # Use distance from cand to all unvisited except self, pick min.
            if n_cand > 1:
                other_unvisited = np.delete(unvisited_nodes, idx)
                d_cand_others = distance_matrix[cand, other_unvisited]
                best_next_cost = d_cand_others.min()
                # Lookahead score: immediate + fraction of next step
                lookahead_score = dists_from_current[idx] + 0.3 * best_next_cost
                # Blend: if lookahead is much worse, increase score
                # Use a soft penalty
                scores[idx] = 0.7 * scores[idx] + 0.3 * (dists_from_current[idx] + 0.5 * best_next_cost)

    _prev_node = current_node
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
