<<<<<<< SEARCH
"""Fresh design: bitmask-accelerated greedy covering with a hard deadline + repair."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin well under 120s
    rng = random.Random(12345)

    # Precompute, for every k-subset mask, the list of t-subset masks it covers.
    masks_by_mask = {}
    for comb in itertools.combinations(range(v), k):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(sm)
        masks_by_mask[mask] = subs

    # Enumerate all t-subset masks in a fixed order (used for seeding/repair).
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        sm = 0
        for p in sub:
            sm |= (1 << p)
        tsub_masks.append(sm)

    # cover_count[i] = how many blocks cover tsub_masks[i]; uncovered = count==0
    cover_count = np.zeros(len(tsub_masks), dtype=np.int32)
    uncovered = cover_count == 0

    # map tsub-mask -> index for quick lookups
    idx_of = {m: i for i, m in enumerate(tsub_masks)}

    blocks = []
    blocks_set = set()
    n = v

    def deadline_hit():
        return time.time() > deadline

    def add_block(mask):
        blocks.append(mask)
        blocks_set.add(mask)
        for sm in masks_by_mask[mask]:
            cover_count[idx_of[sm]] += 1
        uncovered[:] = cover_count == 0
        return uncovered.sum()

    # Greedy: repeatedly open a block on the first uncovered t-subset, then grow it.
    while uncovered.any() and not deadline_hit():
        # seed: smallest uncovered t-subset (deterministic)
        first = int(np.argmax(uncovered))
        seed_mask = tsub_masks[first]
        block_mask = seed_mask
        # points already in block
        in_block = [p for p in range(n) if (block_mask >> p) & 1]
        # covered-so-far t-subset masks (set) for marginal gain
        covered = set(masks_by_mask.get(block_mask, [seed_mask]))
        # ensure seed covers its own t-subsets
        for sm in masks_by_mask.get(block_mask, []):
            covered.add(sm)

        while len(in_block) < k and not deadline_hit():
            best_p = None
            best_gain = -1
            best_div = -1
            for p in range(n):
                if (block_mask >> p) & 1:
                    continue
                # marginal gain: new t-subsets formed by adding p
                gain = 0
                # t-subsets containing p and (t-1) from current block
                for rest in itertools.combinations(in_block, t - 1):
                    sm = 0
                    for q in rest:
                        sm |= (1 << q)
                    sm |= (1 << p)
                    if cover_count[idx_of[sm]] == 0:
                        gain += 1
                if gain < best_gain:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_div = gain
                else:
                    # tie-break: prefer point with fewer existing blocks containing it (diversity)
                    div = 0
                    for b in blocks:
                        if (b >> p) & 1:
                            div += 1
                    if div < best_div:
                        best_div = div
                        best_p = p
            if best_p is None or best_gain <= 0:
                # no beneficial point; fill randomly to reach k (keeps progress)
                candidates = [p for p in range(n) if not ((block_mask >> p) & 1)]
                if not candidates:
                    break
                if len(candidates) > (k - len(in_block)):
                    candidates = rng.sample(candidates, k - len(in_block))
                for p in candidates:
                    block_mask |= (1 << p)
                    in_block.append(p)
                break
            block_mask |= (1 << best_p)
            in_block.append(best_p)
            # update covered set (for potential future use)
            for rest in itertools.combinations(in_block[:-1], t - 1):
                sm = 0
                for q in rest:
                    sm |= (1 << q)
                sm |= (1 << best_p)
                covered.add(sm)

        if block_mask not in blocks_set:
            add_block(block_mask)

    # REPAIR: guarantee validity even if greedy stopped early.
    # For each uncovered t-subset, open a block covering it (deterministic, fast).
    uncovered_idx = np.where(cover_count == 0)[0]
    for i in uncovered_idx:
        if deadline_hit():
            # even if we run out, we MUST cover; do a minimal fill with no time check
            pass
        sm = tsub_masks[i]
        # build a k-subset containing this t-subset
        have = [p for p in range(n) if (sm >> p) & 1]
        need = k - len(have)
        extra = []
        for p in range(n):
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
        if mask not in blocks_set:
            add_block(mask)

    # Convert masks back to sorted lists of points.
    result = []
    for mask in blocks:
        pts = [p for p in range(n) if (mask >> p) & 1]
        pts.sort()
        result.append(pts)
    return result
# EVOLVE-BLOCK-END
=======
"""Vectorized greedy covering design C(19,7,4) with local-search polish."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 108.0  # safety margin under 120s
    rng = random.Random(12345)
    n = v

    # Precompute all t-subset masks and their indices
    tsub_list = list(itertools.combinations(range(n), t))
    tsub_masks = np.array([sum(1 << p for p in sub) for sub in tsub_list], dtype=np.int64)
    n_tsubs = len(tsub_list)
    idx_of = {int(m): i for i, m in enumerate(tsub_masks)}

    # Precompute for each point p: which t-subset indices contain p
    # tsub_has[p, j] = 1 if tsub_list[j] contains point p
    tsub_has = np.zeros((n, n_tsubs), dtype=np.int8)
    for j, sub in enumerate(tsub_list):
        for p in sub:
            tsub_has[p, j] = 1

    # cover_count[j] = number of blocks covering tsub_list[j]
    cover_count = np.zeros(n_tsubs, dtype=np.int32)
    uncovered = np.ones(n_tsubs, dtype=bool)

    blocks = []       # list of point-lists
    blocks_set = set()
    block_masks = []  # list of int masks

    def deadline_hit():
        return time.time() > deadline

    def mask_to_tuple(mask):
        return tuple(p for p in range(n) if (mask >> p) & 1)

    def add_block_from_mask(mask):
        """Add a block given its bitmask; update cover_count."""
        pts = mask_to_tuple(mask)
        blocks_set.add(pts)
        blocks.append(list(pts))
        block_masks.append(mask)
        # Update cover_count: for each t-subset contained in this block, increment
        for sub in itertools.combinations(pts, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            j = idx_of[sm]
            cover_count[j] += 1
        uncovered[:] = (cover_count == 0)

    def add_block_from_points(pts):
        """Add a block given a list of points; update cover_count."""
        pts = tuple(sorted(pts))
        if pts in blocks_set:
            return
        mask = 0
        for p in pts:
            mask |= (1 << p)
        add_block_from_mask(mask)

    # Precompute: for a given set of points S (as bitmask) and candidate point p,
    # compute the gain (number of newly covered t-subsets) vectorized.
    def compute_gain_vectorized(block_pts, candidate_p):
        """
        Given current block points and a candidate point not in block,
        compute how many currently-uncovered t-subsets would be covered
        by adding candidate_p to the block.
        A t-subset {candidate_p} ∪ {t-1 from block_pts} is covered.
        """
        if len(block_pts) < t - 1:
            return 0
        # Generate all (t-1)-subsets of block_pts
        # For each, form the t-subset with candidate_p, check if uncovered
        gains = 0
        for rest in itertools.combinations(block_pts, t - 1):
            sm = (1 << candidate_p)
            for q in rest:
                sm |= (1 << q)
            j = idx_of.get(sm, -1)
            if j >= 0 and cover_count[j] == 0:
                gains += 1
        return gains

    # Greedy phase: build blocks one at a time
    while uncovered.any() and not deadline_hit():
        # Pick the uncovered t-subset with highest "difficulty" (fewest blocks can cover it)
        # For simplicity, pick the first uncovered one (deterministic)
        first_uncovered = int(np.argmax(uncovered))
        seed_tsub = tsub_list[first_uncovered]
        seed_mask = int(tsub_masks[first_uncovered])

        # Start block with the seed t-subset points
        in_block = list(seed_tsub)
        block_mask = seed_mask

        # Greedily add points to maximize marginal gain
        while len(in_block) < k and not deadline_hit():
            candidates = [p for p in range(n) if (block_mask >> p) & 1 == 0]
            if not candidates:
                break

            # Compute gain for each candidate
            best_p = None
            best_gain = -1
            best_div = float('inf')

            for p in candidates:
                gain = compute_gain_vectorized(in_block, p)
                if gain < best_gain:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    best_p = p
                    best_div = gain
                else:
                    # Tie-break: prefer point appearing in fewer existing blocks
                    div = sum(1 for b in blocks if p in b)
                    if div < best_div:
                        best_div = div
                        best_p = p

            if best_p is None or best_gain <= 0:
                # No beneficial addition; fill remaining slots randomly
                remaining = k - len(in_block)
                if remaining > 0:
                    fill = rng.sample(candidates, min(remaining, len(candidates)))
                    for p in fill:
                        in_block.append(p)
                        block_mask |= (1 << p)
                break

            in_block.append(best_p)
            block_mask |= (1 << best_p)

        # Add the completed block
        if block_mask not in blocks_set:
            add_block_from_mask(block_mask)

    # Local search polish: try to improve by swapping points in blocks
    # A swap: remove point a from block, add point b (not in block).
    # This is beneficial if it covers new t-subsets without uncovering any.
    if not deadline_hit():
        # Precompute for each block: set of t-subset indices it covers
        block_tsub_indices = []
        for pts in blocks:
            subs = set()
            for sub in itertools.combinations(pts, t):
                sm = 0
                for p in sub:
                    sm |= (1 << p)
                subs.add(idx_of[sm])
            block_tsub_indices.append(subs)

        # Try swaps for a limited number of iterations
        max_swaps = 200
        swap_count = 0
        improved = True
        while improved and swap_count < max_swaps and not deadline_hit():
            improved = False
            for bi in range(len(blocks)):
                if deadline_hit():
                    break
                pts = blocks[bi]
                mask = block_masks[bi]
                covered_subs = block_tsub_indices[bi]

                for ai, pa in enumerate(pts):
                    if deadline_hit():
                        break
                    for pb in range(n):
                        if pb == pa or (mask >> pb) & 1:
                            continue
                        # Try swap: remove pa, add pb
                        new_pts = [p for i, p in enumerate(pts) if i != ai] + [pb]
                        new_mask = (mask & ~(1 << pa)) | (1 << pb)
                        new_pts_tuple = tuple(sorted(new_pts))
                        if new_pts_tuple in blocks_set:
                            continue

                        # Compute new covered t-subsets
                        new_subs = set()
                        for sub in itertools.combinations(new_pts, t):
                            sm = 0
                            for p in sub:
                                sm |= (1 << p)
                            new_subs.add(idx_of[sm])

                        # Check: does new block cover any t-subset not covered by old?
                        # And does it uncover any t-subset that was only covered by this block?
                        newly_covered = new_subs - covered_subs
                        lost = covered_subs - new_subs

                        # For lost subs, check if still covered by other blocks
                        truly_lost = 0
                        for j in lost:
                            if cover_count[j] <= 1:
                                truly_lost += 1

                        if truly_lost == 0 and len(newly_covered) > 0:
                            # Perform swap
                            for j in lost:
                                cover_count[j] -= 1
                            for j in newly_covered:
                                cover_count[j] += 1
                            uncovered[:] = (cover_count == 0)

                            blocks[bi] = new_pts
                            block_masks[bi] = new_mask
                            blocks_set.remove(tuple(pts))
                            blocks_set.add(new_pts_tuple)
                            block_tsub_indices[bi] = new_subs
                            improved = True
                            swap_count += 1
                            break
                    if improved:
                        break

    # REPAIR: guarantee validity
    uncovered_idx = np.where(cover_count == 0)[0]
    for i in uncovered_idx:
        sm = int(tsub_masks[i])
        have = [p for p in range(n) if (sm >> p) & 1]
        need = k - len(have)
        extra = []
        for p in range(n):
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
        pts_tuple = tuple(block)
        if pts_tuple not in blocks_set:
            add_block_from_mask(mask)

    # Final verification and cleanup
    # Ensure no duplicate blocks
    seen = set()
    final_blocks = []
    for b in blocks:
        t = tuple(b)
        if t not in seen:
            seen.add(t)
            final_blocks.append(b)

    return final_blocks
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
