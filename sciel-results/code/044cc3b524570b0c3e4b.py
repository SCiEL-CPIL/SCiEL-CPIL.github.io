"""Covering design C(19,8,5) - sparse matrix vectorized greedy with redundancy removal."""
import itertools
import time
import random
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)

    # Map each t-subset to an index
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # Build sparse incidence matrix: A[block_idx, subset_idx] = 1 if subset in block
    # We'll build it incrementally as blocks are added.
    # For vectorized gain: gain for adding point x to partial block P =
    #   sum over (t-1)-subsets R of P: A_new[x, R] where A_new[x, R] = 1 if R∪{x} uncovered
    #
    # Strategy: Precompute for each point x, a matrix M_x of shape (C(v-1, t-1), n_tsubsets)
    # where M_x[i, j] = 1 if t-subset j = R_i ∪ {x} (R_i is the i-th (t-1)-subset not containing x)
    #
    # Actually simpler: for each point x, precompute the list of t-subset indices that contain x.
    # Then gain(x, partial_block) = |{ts in tsubsets_containing(x) : ts ⊆ block∪{x} and coverage[ts]==0}|
    #
    # Even better: precompute for each point x a boolean vector of length n_tsubsets
    # indicating which t-subsets contain x. Then:
    #   candidate_tsubsets_for_x = tsubsets containing x that are subsets of (block ∪ {x})
    #   gain = sum(coverage[candidate_tsubsets_for_x] == 0)
    #
    # But computing "subsets of (block ∪ {x})" is still expensive.
    #
    # Better approach: Precompute for each point x and each (t-1)-subset R (not containing x):
    #   the t-subset index of R ∪ {x}
    # Then gain(x, partial_block) = count of R ⊆ partial_block (|R|=t-1) such that coverage[idx(R∪{x})] == 0
    #
    # Vectorized: For each x, precompute a list of (R_tuple, idx) pairs.
    # For a given partial_block, filter to R ⊆ partial_block, then count uncovered.

    # Precompute: for each point x, list of (frozen_set_of_R, idx) where R is (t-1)-subset not containing x
    point_R_info = []
    for x in range(v):
        others = [p for p in range(v) if p != x]
        entries = []
        for R in itertools.combinations(others, t - 1):
            ts = tuple(sorted(R + (x,)))
            idx = tsubset_to_idx[ts]
            entries.append((frozenset(R), idx))
        point_R_info.append(entries)

    # Coverage array
    coverage = np.zeros(n_tsubsets, dtype=np.int32)

    def compute_gain_vectorized(partial_block_set, x):
        """Count uncovered t-subsets covered by adding x to partial_block."""
        if x in partial_block_set:
            return 0
        # Filter R entries where R ⊆ partial_block_set
        entries = point_R_info[x]
        # Vectorized: check which R are subsets of partial_block_set
        # partial_block_set is a set of ints
        count = 0
        for R_fs, idx in entries:
            if R_fs.issubset(partial_block_set) and coverage[idx] == 0:
                count += 1
        return count

    def compute_gain_fast(partial_block_set, x):
        """Faster gain using precomputed bitmasks."""
        # We'll use a different approach below
        pass

    # For speed, precompute bitmask representation
    # Each point gets a bit in an integer (v=19 fits in 64 bits)
    # Each (t-1)-subset R gets a bitmask
    # For each x, store list of (R_mask, idx)
    point_R_masks = []
    for x in range(v):
        others = [p for p in range(v) if p != x]
        entries = []
        for R in itertools.combinations(others, t - 1):
            mask = 0
            for p in R:
                mask |= (1 << p)
            ts = tuple(sorted(R + (x,)))
            idx = tsubset_to_idx[ts]
            entries.append((mask, idx))
        point_R_masks.append(entries)

    def gain_for_x(partial_mask, x):
        """Count uncovered t-subsets that would be newly covered by adding x."""
        if (1 << x) & partial_mask:
            return 0
        count = 0
        for R_mask, idx in point_R_masks[x]:
            if (R_mask & partial_mask) == R_mask and coverage[idx] == 0:
                count += 1
        return count

    def add_block(block):
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            coverage[idx] += 1

    def remove_block(block):
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            coverage[idx] -= 1

    blocks = []
    block_masks = []  # bitmask for each block

    # Greedy construction
    while True:
        if time.time() > deadline - 35.0:
            break
        if np.all(coverage > 0):
            break

        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            break
        start_idx = uncovered_indices[0]
        start_subset = all_tsubsets[start_idx]

        block = list(start_subset)
        block_mask = 0
        for p in block:
            block_mask |= (1 << p)

        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if (1 << x) & block_mask:
                    continue
                g = gain_for_x(block_mask, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_mask |= (1 << best_x)

        block.sort()
        blocks.append(block)
        block_masks.append(block_mask)
        add_block(block)

    # Redundancy removal with multiple passes
    def redundancy_removal():
        nonlocal blocks, block_masks, coverage
        changed = True
        while changed and time.time() < deadline - 15.0:
            changed = False
            i = 0
            while i < len(blocks):
                if time.time() > deadline - 15.0:
                    break
                block = blocks[i]
                is_redundant = True
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx[ts]
                    if coverage[idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    remove_block(block)
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                else:
                    i += 1

    redundancy_removal()

    # Second phase: try to find blocks that cover multiple uncovered subsets
    # Use randomized greedy with different starting points
    rng = random.Random(42)
    while time.time() < deadline - 20.0:
        if np.all(coverage > 0):
            break
        uncovered_indices = np.where(coverage == 0)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick a random uncovered t-subset
        start_idx = uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)]
        start_subset = all_tsubsets[start_idx]

        block = list(start_subset)
        block_mask = 0
        for p in block:
            block_mask |= (1 << p)

        while len(block) < k:
            best_x = None
            best_gain = -1
            for x in range(v):
                if (1 << x) & block_mask:
                    continue
                g = gain_for_x(block_mask, x)
                if g > best_gain:
                    best_gain = g
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_mask |= (1 << best_x)

        block.sort()
        # Check if this block is already present
        block_tuple = tuple(block)
        if block_tuple not in [tuple(b) for b in blocks]:
            blocks.append(block)
            block_masks.append(block_mask)
            add_block(block)

    # Final redundancy removal
    if time.time() < deadline - 5.0:
        redundancy_removal()

    return blocks
