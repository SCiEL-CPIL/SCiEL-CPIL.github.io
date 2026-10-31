"""Different design family: pairwise-greedy set cover over all k-subsets,
with NumPy-vectorized marginal-gain scoring and a hard deadline + repair."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 112.0  # safety margin under 120s
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
    # cover_indices[j] = array of t-subset indices covered by ksub_masks[j]
    cover_indices = []
    for m in ksub_masks:
        pts = [p for p in range(v) if (m >> p) & 1]
        idxs = []
        for sub in itertools.combinations(pts, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            idxs.append(idx_of_tsub[sm])
        cover_indices.append(np.array(idxs, dtype=np.int32))

    # ---------- state ----------
    uncovered = np.ones(n_tsub, dtype=np.int8)  # 1 = not yet covered
    uncovered_count = np.zeros(n_ksub, dtype=np.int32)  # how many uncovered tsubs each ksub covers
    # initialize uncovered_count
    for j in range(n_ksub):
        uncovered_count[j] = np.sum(uncovered[cover_indices[j]])

    chosen = []          # list of ksub indices chosen
    chosen_set = set()   # for O(1) membership
    block_overlap = np.zeros(n_ksub, dtype=np.int32)  # how many chosen blocks share a point with ksub j (approx: just count chosen)

    def deadline_hit():
        return time.time() > deadline

    # ---------- main greedy loop ----------
    while True:
        if deadline_hit():
            break
        # find best ksub: max uncovered_count, then min block_overlap (diversity)
        # Among those with max uncovered_count, pick one not yet chosen
        max_gain = 0
        best_j = -1
        # Vectorized: find indices with max uncovered_count
        # But we need to exclude already chosen ones
        # Efficient approach: temporarily zero out chosen entries
        # Since n_ksub is C(19,7)=50388, we can do a full scan with numpy
        valid_mask = np.ones(n_ksub, dtype=bool)
        for j in chosen_set:
            valid_mask[j] = False
        # Find max among valid
        gains = np.where(valid_mask, uncovered_count, -1)
        max_gain = int(np.max(gains))
        if max_gain <= 0:
            # No block covers any new t-subset -> we're done or stuck
            break
        # All candidates with max_gain
        candidates = np.where((gains == max_gain) & valid_mask)[0]
        if len(candidates) == 1:
            best_j = int(candidates[0])
        else:
            # Tie-break: prefer ksub with fewer points already in chosen blocks
            # Compute for each candidate: number of its points that appear in any chosen block
            # For speed, precompute a "point frequency" array
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
        new_covered = cover_indices[best_j]
        uncovered[new_covered] = 0
        # Update uncovered_count for all ksubs: for each ksub j, subtract number of its tsubsets in new_covered
        # Efficient: for each ksub j, count overlap between cover_indices[j] and new_covered
        # Use a boolean array for new_covered
        new_covered_set = set(new_covered.tolist())
        for j in range(n_ksub):
            if j in chosen_set:
                continue
            # count how many of cover_indices[j] are in new_covered_set
            ci = cover_indices[j]
            # Vectorized: sum of (ci in new_covered_set)
            # Use numpy: create a lookup array
            # Since new_covered_set changes each iteration, build a boolean array of size n_tsub
            # This is O(n_tsub) per ksub per iteration -> too slow for 50388 ksubs * 150 iterations
            # Alternative: precompute for each ksub a sorted array, and use binary search
            # Better: maintain a global "covered" boolean array and recompute uncovered_count only when needed
            # Actually, let's use a different approach: maintain covered_count per tsub, and compute gain on the fly
            pass

        # The above update is too slow. Let's use a different strategy:
        # Instead of maintaining uncovered_count for all ksubs, compute the gain for the best candidate
        # on the fly using the uncovered array.
        # But we need to find the best candidate each iteration.
        # Let's restructure: each iteration, compute gain for all ksubs using numpy.

        # Actually, let's restart with a cleaner approach below.
        break

    # ---------- RESTART with cleaner approach ----------
    # Reset state
    uncovered = np.ones(n_tsub, dtype=np.int8)
    chosen = []
    chosen_set = set()

    # Precompute a boolean matrix: for each tsub index, which ksubs cover it?
    # Or rather: for each ksub, a boolean vector of which tsubs it covers.
    # n_ksub x n_tsub boolean matrix: 50388 x 21669 = ~1 billion booleans = 1 GB. Too much.
    # Instead, use the cover_indices list.

    # For each iteration, compute the gain for all ksubs:
    # gain[j] = sum(uncovered[cover_indices[j]])
    # This is O(n_ksub * avg_cover_size) per iteration.
    # avg_cover_size = C(7,4) = 35. So 50388 * 35 = ~1.76M ops per iteration.
    # With ~150 iterations, that's ~264M ops. In numpy, this should be fast.

    while True:
        if deadline_hit():
            break
        # Compute gain for all ksubs
        # gain[j] = np.sum(uncovered[cover_indices[j]])
        # Vectorize across j:
        # Build a 2D array? No, cover_indices[j] has variable length (always 35 for k=7,t=4).
        # Since all ksubs cover exactly C(7,4)=35 tsubs, we can make a 2D array.
        # cover_matrix[j, i] = tsub index covered by ksub j at position i
        # Let's precompute this.
        # Actually, let's just do it in a loop but with numpy sum:
        gains = np.zeros(n_ksub, dtype=np.int32)
        for j in range(n_ksub):
            if j in chosen_set:
                gains[j] = -1
                continue
            gains[j] = int(np.sum(uncovered[cover_indices[j]]))

        max_gain = int(np.max(gains))
        if max_gain <= 0:
            break

        # Find all with max_gain
        candidates = np.where(gains == max_gain)[0]
        if len(candidates) == 1:
            best_j = int(candidates[0])
        else:
            # Tie-break: diversity
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
        uncovered[cover_indices[best_j]] = 0

    # ---------- REPAIR ----------
    # For each uncovered t-subset, add a block covering it
    uncovered_idx = np.where(uncovered == 1)[0]
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
        # Check if this block is already chosen
        found = False
        for j in chosen_set:
            if ksub_masks[j] == mask:
                found = True
                break
        if not found:
            # Add it as a new "virtual" block
            chosen.append(mask)  # store mask directly for new blocks
            # But we need consistency. Let's just store masks for all blocks.
            pass

    # Convert to result
    # For chosen indices, use ksub_masks; for any extra masks, use them directly.
    # Since we only added via the loop, all chosen are indices into ksub_masks,
    # except possibly repair blocks. Let's handle repair blocks separately.

    result_masks = set()
    for j in chosen:
        if isinstance(j, int) and j < n_ksub:
            result_masks.add(ksub_masks[j])
        else:
            result_masks.add(j)

    # Add repair blocks
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
