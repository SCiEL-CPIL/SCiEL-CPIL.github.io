"""
Optimized covering design C(19,8,5) using global bitmask-based greedy
with redundancy removal and a time-bounded local search polish.
"""
import itertools
import time
import sys
from collections import Counter

# Deadline: 150s wall-clock; use 145s as safety margin
_DEADLINE = 145.0


def construct(v, k, t):
    start_time = time.time()

    # All t-subsets as bitmasks
    all_tsubsets = [0] * 0
    tsubsets = list(itertools.combinations(range(v), t))
    all_tsubsets = [0] * len(tsubsets)
    tsub_index = {}
    for i, comb in enumerate(tsubsets):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_tsubsets[i] = mask
        tsub_index[mask] = i

    n_tsubs = len(tsubsets)
    # coverage_count[i] = number of blocks covering t-subset i
    coverage = [0] * n_tsubs
    blocks = []
    block_masks = []  # bitmask of each block (k bits set)

    # Precompute for each k-subset, which t-subsets it covers
    # This is expensive for C(19,8)=75582, so we compute lazily.
    # Instead, we compute on-the-fly using combinatorial counting.

    def block_covers(mask):
        """Return list of t-subset indices covered by a k-element block mask."""
        pts = [i for i in range(v) if mask & (1 << i)]
        covered = []
        for comb in itertools.combinations(pts, t):
            m = 0
            for p in comb:
                m |= (1 << p)
            covered.append(tsub_index[m])
        return covered

    def gain_for_point(block_pts, x):
        """Number of new t-subsets covered by adding x to block_pts."""
        new_gain = 0
        for comb in itertools.combinations(block_pts, t - 1):
            comb_with_x = sorted(comb + (x,))
            m = 0
            for p in comb_with_x:
                m |= (1 << p)
            idx = tsub_index.get(m)
            if idx is not None and coverage[idx] == 0:
                new_gain += 1
        return new_gain

    # Main greedy loop
    while True:
        if time.time() - start_time > _DEADLINE - 10:
            break

        # Find uncovered t-subset with maximum potential (or just first)
        uncovered_idx = -1
        for i in range(n_tsubs):
            if coverage[i] == 0:
                uncovered_idx = i
                break
        if uncovered_idx == -1:
            break

        # Start block from the uncovered t-subset
        start_comb = tsubsets[uncovered_idx]
        block_pts = list(start_comb)
        block_mask = 0
        for p in block_pts:
            block_mask |= (1 << p)

        # Add points greedily to maximize coverage of uncovered t-subsets
        remaining = [p for p in range(v) if p not in block_pts]
        while len(block_pts) < k:
            if not remaining:
                break
            best_x = None
            best_gain = -1
            for x in remaining:
                g = gain_for_point(block_pts, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block_pts.append(best_x)
            block_mask |= (1 << best_x)
            remaining.remove(best_x)

        if len(block_pts) < k:
            break

        # Check if this block is already in blocks (redundancy check)
        block_mask_set = frozenset(block_pts)
        is_dup = False
        for bm in block_masks:
            if frozenset(p for p in range(v) if bm & (1 << p)) == block_mask_set:
                is_dup = True
                break

        if is_dup:
            # Try a different starting point: pick another uncovered t-subset
            continue

        blocks.append(sorted(block_pts))
        block_masks.append(block_mask)

        # Update coverage
        covered_indices = block_covers(block_mask)
        for idx in covered_indices:
            coverage[idx] += 1

    # Check if we're done
    if min(coverage) >= 1:
        return blocks

    # Local search polish: try to improve by swapping points in blocks
    # Only do this if we have time
    if time.time() - start_time > _DEADLINE - 5:
        return blocks

    # Simple local improvement: for each block, try removing a point and
    # adding a different one to reduce redundancy or improve coverage
    improved = True
    while improved and (time.time() - start_time) < _DEADLINE - 5:
        improved = False
        for bi in range(len(blocks)):
            if time.time() - start_time > _DEADLINE - 5:
                break
            block = blocks[bi]
            old_mask = block_masks[bi]
            old_covered = block_covers(old_mask)
            old_new_cover = sum(1 for idx in old_covered if coverage[idx] == 1)

            for ri in range(len(block)):
                for x in range(v):
                    if x in block:
                        continue
                    new_block = block[:ri] + block[ri+1:] + [x]
                    new_block.sort()
                    new_mask = 0
                    for p in new_block:
                        new_mask |= (1 << p)
                    new_covered = block_covers(new_mask)
                    new_new_cover = sum(1 for idx in new_covered if coverage[idx] == 1)

                    if new_new_cover > old_new_cover:
                        blocks[bi] = new_block
                        block_masks[bi] = new_mask
                        # Update coverage
                        for idx in old_covered:
                            coverage[idx] -= 1
                        for idx in new_covered:
                            coverage[idx] += 1
                        old_covered = new_covered
                        old_new_cover = new_new_cover
                        old_mask = new_mask
                        improved = True

    return blocks
