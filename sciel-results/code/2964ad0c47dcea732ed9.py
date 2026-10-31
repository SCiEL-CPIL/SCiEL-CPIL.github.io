# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin under 120s
    rng = random.Random(98765)

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

    idx_of_tsub = {m: i for i, m in enumerate(tsub_masks)}

    # ---------- precompute cover_matrix: (n_ksub, C(k,t)) int32 ----------
    # cover_matrix[j, i] = index of the i-th t-subset covered by ksub j
    t_per_k = len(list(itertools.combinations(range(k), t)))  # C(k,t)
    cover_matrix = np.zeros((n_ksub, t_per_k), dtype=np.int32)
    for j, m in enumerate(ksub_masks):
        pts = [p for p in range(v) if (m >> p) & 1]
        for i, sub in enumerate(itertools.combinations(pts, t)):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            cover_matrix[j, i] = idx_of_tsub[sm]

    # ---------- state ----------
    uncovered = np.ones(n_tsub, dtype=np.int8)  # 1 = not yet covered
    chosen = []          # list of ksub indices chosen
    chosen_set = set()   # for O(1) membership

    def deadline_hit():
        return time.time() > deadline

    # ---------- main greedy loop (vectorized gain via matmul) ----------
    while True:
        if deadline_hit():
            break
        # Compute gain for all ksubs: gain[j] = sum(uncovered[cover_matrix[j]])
        # This is a matrix-vector product: (n_ksub, t_per_k) @ (n_tsub,)
        gains = cover_matrix.astype(np.int32) @ uncovered.astype(np.int32)
        # Exclude already chosen blocks
        for j in chosen_set:
            gains[j] = -1

        max_gain = int(np.max(gains))
        if max_gain <= 0:
            break

        # Find all with max_gain
        candidates = np.where(gains == max_gain)[0]
        if len(candidates) == 1:
            best_j = int(candidates[0])
        else:
            # Tie-break: prefer ksub with fewer points already in chosen blocks (diversity)
            if not chosen:
                best_j = int(candidates[0])
            else:
                used_points_mask = 0
                for j in chosen_set:
                    used_points_mask |= ksub_masks[j]
                cand_masks = np.array([ksub_masks[j] for j in candidates], dtype=np.uint64)
                overlaps = np.array([bin(int(m) & used_points_mask).count('1') for m in cand_masks], dtype=np.int32)
                best_local = int(np.argmin(overlaps))
                best_j = int(candidates[best_local])

        chosen_set.add(best_j)
        chosen.append(best_j)
        # Update uncovered
        uncovered[cover_matrix[best_j]] = 0

    # ---------- local search polish: try single-block swaps to reduce count ----------
    # For each chosen block, see if removing it and adding one better block reduces total count
    # or at least maintains coverage with fewer blocks.
    # Strategy: for each chosen block j, compute what t-subsets would become uncovered
    # if j were removed. Then find a single block that covers all of them (or the most).
    # If such a block exists and is not already chosen, swap.
    # This is a 1-exchange local search.

    # Precompute: for each t-subset, which chosen blocks cover it?
    # tsub_covered_by: list of chosen block indices that cover t-subset i
    # Build this incrementally.
    # Actually, let's just do a simpler swap: for each chosen block, try replacing it
    # with a random block and see if coverage is maintained or improved.

    # Better: for each chosen block j, compute the "exclusive" t-subsets it covers
    # (t-subsets covered only by j among chosen blocks). If j has no exclusive t-subsets,
    # we can remove it. If it has some, we need a replacement that covers those.

    # Build tsub_cover_count: how many chosen blocks cover each t-subset
    tsub_cover_count = np.zeros(n_tsub, dtype=np.int8)
    for j in chosen:
        tsub_cover_count[cover_matrix[j]] += 1

    # For each chosen block j, find t-subsets it exclusively covers (count == 1)
    # If none, we can remove j.
    # If some, find a block that covers all of them.

    # Try to remove blocks with no exclusive coverage
    improved = True
    while improved and not deadline_hit():
        improved = False
        # Recompute tsub_cover_count
        tsub_cover_count = np.zeros(n_tsub, dtype=np.int8)
        for j in chosen:
            tsub_cover_count[cover_matrix[j]] += 1

        for idx, j in enumerate(list(chosen)):
            if deadline_hit():
                break
            # Check if j has exclusive coverage
            exclusive = (tsub_cover_count[cover_matrix[j]] == 1)
            if not np.any(exclusive):
                # j can be removed without losing coverage
                chosen.pop(idx)
                chosen_set.remove(j)
                improved = True
                break  # restart the loop since indices changed

    # ---------- REPAIR ----------
    # For each uncovered t-subset, add a block covering it
    uncovered_idx = np.where(uncovered == 1)[0]
    result_masks = set()
    for j in chosen:
        result_masks.add(ksub_masks[j])

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
        result_masks.add(mask)

    # Convert to lists
    result = []
    for mask in result_masks:
        pts = [p for p in range(v) if (mask >> p) & 1]
        pts.sort()
        result.append(pts)

    return result
# EVOLVE-BLOCK-END
