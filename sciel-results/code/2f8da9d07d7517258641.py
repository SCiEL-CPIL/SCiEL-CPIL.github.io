# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 115.0  # safety margin under 120s
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

    # map t-subset mask -> index
    idx_of_tsub = {m: i for i, m in enumerate(tsub_masks)}

    # ---------- precompute: for each k-subset, the indices of t-subsets it covers ----------
    # Since every k-subset covers exactly C(k, t) t-subsets, we can use a dense 2D array.
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
    uncovered = np.ones(n_tsub, dtype=bool)  # True = not yet covered
    chosen = []          # list of ksub indices chosen
    chosen_set = set()   # for O(1) membership

    def deadline_hit():
        return time.time() > deadline

    # ---------- main greedy loop ----------
    while True:
        if deadline_hit():
            break
        
        # Compute gain for all ksubs: gain[j] = number of uncovered tsubs covered by ksub j
        # Vectorized: take uncovered values at the indices in cover_matrix
        # cover_matrix is (n_ksub, c_k_t)
        # uncovered is (n_tsub,)
        # np.take(uncovered, cover_matrix) -> (n_ksub, c_k_t) boolean array
        # sum along axis 1 -> (n_ksub,)
        gains = np.take(uncovered, cover_matrix).sum(axis=1)
        
        # Exclude already chosen blocks
        for j in chosen_set:
            gains[j] = -1
            
        max_gain = int(np.max(gains))
        if max_gain <= 0:
            break
        
        # Find all candidates with max_gain
        candidates = np.where(gains == max_gain)[0]
        
        if len(candidates) == 1:
            best_j = int(candidates[0])
        else:
            # Tie-break: prefer ksub with fewer points already in chosen blocks (diversity)
            if not chosen:
                best_j = int(candidates[0])
            else:
                # Build a bitmask of all points used in chosen blocks
                used_points_mask = 0
                for j in chosen_set:
                    used_points_mask |= ksub_masks[j]
                
                # For each candidate, count how many of its points are in used_points_mask
                # Vectorize: for each candidate j, popcount(ksub_masks[j] & used_points_mask)
                cand_masks = np.array([ksub_masks[j] for j in candidates], dtype=np.uint64)
                overlaps = np.array([bin(int(m) & used_points_mask).count('1') for m in cand_masks], dtype=np.int32)
                best_local = int(np.argmin(overlaps))
                best_j = int(candidates[best_local])

        # Commit best_j
        chosen_set.add(best_j)
        chosen.append(best_j)
        
        # Update uncovered: zero out tsubsets covered by best_j
        uncovered[cover_matrix[best_j, :]] = False

    # ---------- REPAIR ----------
    # For each uncovered t-subset, add a block covering it
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
