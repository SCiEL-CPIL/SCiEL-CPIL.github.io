"""Constructive TSP heuristic: phase-aware NN blended with destination proximity."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard: the evaluator kills after 150 s wall-clock.
# We track a start time on first call and fall back to a cheap deterministic
# choice if we are within the safety margin of the limit.
_T0 = None
_DEADLINE = 150.0
_SAFETY = 5.0  # seconds of margin before we force the fast path


def _elapsed():
    global _T0
    if _T0 is None:
        _T0 = time.time()
    return time.time() - _T0


def _safe_index(unvisited_nodes, idx):
    # Guarantee we return a valid node id even under unexpected input.
    try:
        arr = np.asarray(unvisited_nodes)
        if arr.size == 0:
            return 0
        if not (0 <= idx < arr.size):
            idx = 0
        return int(arr[idx])
    except Exception:
        return 0


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    unvisited_nodes = np.asarray(unvisited_nodes)
    n = unvisited_nodes.size
    if n == 0:
        return int(current_node)

    # Hard deadline guard: if we are close to the limit, take the cheap NN pick.
    if _elapsed() > _DEADLINE - _SAFETY:
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])

    try:
        d_cur = distance_matrix[current_node, unvisited_nodes]      # (n,)
        d_dest = distance_matrix[unvisited_nodes, destination_node]  # (n,)

        # Smooth phase ramp: early -> NN dominant, late -> destination dominant.
        # progress in [0,1]; 1.0 means only this pick + closing remain.
        progress = (50 - n) / max(50 - 1, 1)  # 50-city instances
        progress = float(np.clip(progress, 0.0, 1.0))

        # Weights: NN weight decays, destination weight grows.
        # Keep a floor on NN so we never fully ignore local tightness.
        w_nn = 1.0 - 0.85 * progress
        w_dest = 0.85 * progress

        # Normalize each component to [0,1] scale so the blend is stable.
        # Use max (not sum) to keep scale comparable across candidates.
        m_cur = float(np.max(d_cur)) if np.max(d_cur) > 0 else 1.0
        m_dest = float(np.max(d_dest)) if np.max(d_dest) > 0 else 1.0

        score = w_nn * (d_cur / m_cur) + w_dest * (d_dest / m_dest)

        # Cheap tie-breaker: among near-equal scores, prefer the candidate whose
        # best remaining neighbor is also close (reduces future long edges).
        best_idx = int(np.argmin(score))
        threshold = score[best_idx] + 1e-9
        tied = np.where(score <= threshold)[0]
        if tied.size > 1:
            # For each tied candidate, look at its best distance to the other
            # unvisited nodes (excluding itself).
            tied_list = list(tied)
            best_tie = tied_list[0]
            best_tie_val = np.inf
            for t in tied_list:
                node_t = unvisited_nodes[t]
                if n > 1:
                    # distances from node_t to all unvisited except itself
                    sub = np.delete(d_cur, t)  # proxy: use current-based, cheap
                    # better: direct row
                    row = distance_matrix[node_t, unvisited_nodes]
                    row_t = np.delete(row, t)
                    if row_t.size > 0:
                        val = float(np.min(row_t))
                    else:
                        val = 0.0
                else:
                    val = 0.0
                if val < best_tie_val:
                    best_tie_val = val
                    best_tie = t
            best_idx = best_tie

        return int(unvisited_nodes[best_idx])
    except Exception:
        # Last-resort safe fallback.
        try:
            d = distance_matrix[current_node, unvisited_nodes]
            return int(unvisited_nodes[np.argmin(d)])
        except Exception:
            return int(unvisited_nodes[0])
# EVOLVE-BLOCK-END
