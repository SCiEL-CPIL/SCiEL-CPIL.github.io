"""Improved constructive TSP heuristic: 3-step lookahead with regret and destination bias."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # seconds wall-clock deadline (safety margin before 150s kill)
_start_time = time.time()


def _select_blended(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Pick next node using 3-step lookahead with regret and destination awareness."""
    unv = np.asarray(unvisited_nodes)
    n = unv.size
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unv[0])

    # Deadline guard: if close to limit, fall back to simple nearest neighbor
    elapsed = time.time() - _start_time
    if elapsed > _DEADLINE:
        dm = distance_matrix
        d_cur = dm[current_node, unv]
        idx = int(np.argmin(d_cur))
        return int(unv[idx])

    dm = distance_matrix
    # Cost from current to each candidate
    d_cur = dm[current_node, unv]

    # Build submatrix of distances among unvisited nodes
    sub = dm[np.ix_(unv, unv)]
    sub_diag_inf = sub.copy()
    np.fill_diagonal(sub_diag_inf, np.inf)
    
    # 2-step: min onward distance from each candidate to any other unvisited
    d_onward_1 = sub_diag_inf.min(axis=1)
    two_step = d_cur + d_onward_1

    # 3-step lookahead: for each candidate i, find min over j != i of (sub[i,j] + min_{k != i,j} sub[j,k])
    # This is expensive O(n^3), but for n=50 it's manageable
    # Optimize: for each j, precompute min over k != i,j
    # Actually, let's compute 3-step more carefully
    # For candidate i: cost = d_cur[i] + min_{j != i} (sub[i,j] + min_{k != i,j} sub[j,k])
    
    # Precompute for each pair (j, k) where k != j: sub[j,k]
    # For each j, the min over k != i,j is: if i is the argmin of row j (excluding j itself), use second min; else use min
    
    # Let's compute row mins and second mins
    row_min = np.full((n, 2), np.inf)
    for j in range(n):
        row_j = sub_diag_inf[j, :]  # inf at position j
        # Get two smallest values
        sorted_idx = np.argsort(row_j)
        if len(sorted_idx) >= 2:
            row_min[j, 0] = row_j[sorted_idx[0]]
            row_min[j, 1] = row_j[sorted_idx[1]]
    
    # For 3-step: for candidate i, cost_i = d_cur[i] + min_{j != i} (sub[i,j] + min_{k != i,j} sub[j,k])
    # min_{k != i,j} sub[j,k] = row_min[j,0] if argmin is not i, else row_min[j,1]
    # We need argmin indices
    row_argmin = np.full((n, 2), -1, dtype=int)
    for j in range(n):
        row_j = sub_diag_inf[j, :]
        sorted_idx = np.argsort(row_j)
        if len(sorted_idx) >= 2:
            row_argmin[j, 0] = sorted_idx[0]
            row_argmin[j, 1] = sorted_idx[1]
    
    # Now compute 3-step cost
    three_step = np.full(n, np.inf)
    for i in range(n):
        # For each j != i
        best = np.inf
        for j in range(n):
            if j == i:
                continue
            # min_{k != i,j} sub[j,k]
            if row_argmin[j, 0] != i:
                min_k = row_min[j, 0]
            else:
                min_k = row_min[j, 1]
            cost = sub[i, j] + min_k
            if cost < best:
                best = cost
        three_step[i] = d_cur[i] + best

    # Regret-based scoring: use 3-step as primary
    # Destination awareness
    d_dest = dm[unv, destination_node]
    d_dest_range = d_dest.max() - d_dest.min()
    if d_dest_range > 1e-12:
        d_dest_norm = (d_dest - d_dest.min()) / d_dest_range
    else:
        d_dest_norm = np.zeros_like(d_dest)

    # Normalize 3-step
    three_step_span = three_step.max() - three_step.min()
    if three_step_span > 1e-12:
        three_step_norm = (three_step - three_step.min()) / three_step_span
    else:
        three_step_norm = np.zeros_like(three_step)

    # Normalize 2-step
    two_step_span = two_step.max() - two_step.min()
    if two_step_span > 1e-12:
        two_step_norm = (two_step - two_step.min()) / two_step_span
    else:
        two_step_norm = np.zeros_like(two_step)

    # Normalize nearest
    d_cur_span = d_cur.max() - d_cur.min()
    if d_cur_span > 1e-12:
        d_cur_norm = (d_cur - d_cur.min()) / d_cur_span
    else:
        d_cur_norm = np.zeros_like(d_cur)

    # Weighted score: emphasize longer lookahead
    score = (0.60 * three_step_norm + 0.25 * two_step_norm + 0.10 * d_cur_norm + 0.05 * d_dest_norm)

    idx = int(np.argmin(score))
    return int(unv[idx])


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    return _select_blended(current_node, destination_node, unvisited_nodes, distance_matrix)
# EVOLVE-BLOCK-END
