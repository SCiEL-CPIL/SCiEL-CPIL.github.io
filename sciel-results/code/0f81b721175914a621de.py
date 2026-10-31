# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin under 120s

    # ---------- enumerate all t-subsets and all k-subsets as bitmasks ----------
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    n_tsub = len(tsub_masks)

    ksub_masks = []
    for comb in itertools.combinations(range(v), k):
        m = 0
        for p in comb:
            m |= (1 << p)
        ksub_masks.append(m)
    n_ksub = len(ksub_masks)

    # map t-subset mask -> index
    idx_of_tsub = {m: i for i, m in enumerate(tsub_masks)}

    # ---------- precompute: for each k-subset, the indices of t-subsets it covers ----------
    c_k_t = len(list(itertools.combinations(range(k), t)))
    cover_matrix = np.zeros((n_ksub, c_k_t), dtype=np.int32)

    for j, m in enumerate(ksub_masks):
        pts = [p for p in range(v) if (m >> p) & 1]
        idxs = []
        for sub in itertools.combinations(pts, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            idxs.append(idx_of_tsub[sm])
        cover_matrix[j, :] = idxs

    # ---------- state ----------
    uncovered = np.ones(n_tsub, dtype=bool)
    chosen = []
    chosen_set = set()

    # Point usage counts: how many chosen blocks contain each point
    point_usage = np.zeros(v, dtype=np.int32)

    # ---------- deficiency-ratio greedy ----------
    # score_j = gain_j / (1 + sum_{p in block_j} max(0, point_usage[p] - 1))
    # This penalizes blocks that reuse already-saturated points,
    # naturally spreading coverage across the point set.

    def deadline_hit():
        return time.time() > deadline

    # Precompute point membership for each ksub as a list of points
    ksub_points = []
    for m in ksub_masks:
        pts = [p for p in range(v) if (m >> p) & 1]
        ksub_points.append(pts)

    # Precompute for each ksub, the list of points as an array for fast penalty calc
    ksub_points_arr = np.array(ksub_points, dtype=np.int32)  # shape (n_ksub, k)

    while True:
        if deadline_hit():
            break

        # Compute gain for all ksubs
        gains = np.take(uncovered, cover_matrix).sum(axis=1)

        # Exclude already chosen blocks
        for j in chosen_set:
            gains[j] = -1

        max_gain = int(np.max(gains))
        if max_gain <= 0:
            break

        # Compute deficiency ratio for all candidates with positive gain
        # penalty_j = 1 + sum_{p in block_j} max(0, point_usage[p] - 1)
        # Vectorized: for each ksub j, sum over its k points of max(0, point_usage[p]-1)
        # ksub_points_arr has shape (n_ksub, k)
        # point_usage[ksub_points_arr] has shape (n_ksub, k)
        # np.maximum(0, point_usage[ksub_points_arr] - 1) -> (n_ksub, k)
        # sum along axis 1 -> (n_ksub,)
        point_penalties = np.maximum(0, point_usage[ksub_points_arr] - 1).sum(axis=1)
        denominators = 1.0 + point_penalties.astype(np.float64)

        # Score: gain / denominator, with -1 for excluded
        scores = np.full(n_ksub, -1.0, dtype=np.float64)
        valid_mask = gains > 0
        scores[valid_mask] = gains[valid_mask].astype(np.float64) / denominators[valid_mask]

        best_j = int(np.argmax(scores))
        if scores[best_j] <= 0:
            break

        # Commit
        chosen_set.add(best_j)
        chosen.append(best_j)

        # Update uncovered
        uncovered[cover_matrix[best_j, :]] = False

        # Update point usage
        for p in ksub_points[best_j]:
            point_usage[p] += 1

    # ---------- REPAIR ----------
    uncovered_idx = np.where(uncovered)[0]
    repair_blocks = []
    for i in uncovered_idx:
        sm = tsub_masks[i]
        have = [p for p in range(v) if (sm >> p) & 1]
        need = k - len(have)
        extra = []
        for p in range(v):
            if need <= 0:
                break
            if not ((sm >> p) & 1):
                extra.append(p)
                need -= 1
        block = have + extra
        block.sort()
        mask = 0
        for p in block:
            mask |= (1 << p)
        repair_blocks.append(mask)

    # Convert to result
    result_masks = set()
    for j in chosen:
        result_masks.add(ksub_masks[j])

    for mask in repair_blocks:
        result_masks.add(mask)

    # Convert to lists
    result = []
    for mask in result_masks:
        pts = [p for p in range(v) if (mask >> p) & 1]
        pts.sort()
        result.append(pts)

    return result
# EVOLVE-BLOCK-END
