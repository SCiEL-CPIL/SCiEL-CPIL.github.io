"""Global bitmask-based greedy covering with dominance (redundancy) removal.

Covers C(19,8,5): find as few 8-subsets of 19 points as possible so every
5-subset lies in some block.  Uses a bitmask universe (each t-subset is a
v-bit int), a fast uncovered-set, and post-hoc removal of blocks that are
subsets of a later block (their t-subsets are then all covered).
"""
import time
import itertools
import numpy as np


def _build_universe(v, t):
    """Return (all_masks, comb) where comb[i] is the i-th t-subset of range(v)
    as a bitmask, in itertools.combinations order."""
    all_masks = []
    for c in itertools.combinations(range(v), t):
        m = 0
        for x in c:
            m |= 1 << x
        all_masks.append(m)
    all_masks = np.array(all_masks, dtype=np.uint64)
    # Precompute, for each point x, the list of universe indices whose mask
    # contains x.  Used to build a block's t-subset mask list quickly.
    point_to_idx = {}
    for i, m in enumerate(all_masks):
        mm = int(m)
        y = 0
        while mm:
            lsb = mm & (-mm)
            y = lsb.bit_length() - 1
            point_to_idx.setdefault(y, []).append(i)
            mm ^= lsb
    return all_masks, point_to_idx


def _block_tmask_indices(block, point_to_idx, t):
    """Return the list of universe indices (into all_masks) that are t-subsets
    of `block`.  A t-subset of block is exactly a t-subset of the points of
    block, i.e. an element of the universe whose mask is a subset of block's
    point mask."""
    pm = 0
    for x in block:
        pm |= 1 << x
    idx = []
    # iterate over the t-subsets of block directly (cheap: C(k,t) <= 126)
    for c in itertools.combinations(block, t):
        m = 0
        for x in c:
            m |= 1 << x
        idx.append(m)  # store the mask itself; membership set is over masks
    return idx


def construct(v, k, t):
    t0 = time.time()
    deadline = 120.0          # hard wall-clock limit (evaluator kills at 150s)
    safety = 150.0 - 3.0      # return by this much
    deadline = min(deadline, safety)

    # Universe of all t-subsets as bitmasks.
    all_masks_list = []
    for c in itertools.combinations(range(v), t):
        m = 0
        for x in c:
            m |= 1 << x
        all_masks_list.append(m)
    all_masks = np.array(all_masks_list, dtype=np.uint64)
    n = len(all_masks)

    uncovered = set(all_masks_list)
    blocks = []
    points = list(range(v))

    while uncovered:
        if time.time() >= deadline - 2.0:
            break
        # Seed the block with the lexicographically-first uncovered t-subset
        # (deterministic, and it guarantees the block will cover >=1 subset).
        seed_mask = min(uncovered)
        seed = [x for x in range(v) if (seed_mask >> x) & 1]
        block = list(seed)

        while len(block) < k:
            in_block = set(block)
            best_x = None
            best_gain = -1
            for x in points:
                if x in in_block:
                    continue
                # Newly covered subsets = (t-subsets of block+{x}) that are
                # still uncovered.
                gain = 0
                for c in itertools.combinations(block, t - 1):
                    m = 0
                    for y in c:
                        m |= 1 << y
                    m |= 1 << x
                    if m in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                # No further gain; pad with arbitrary points to reach size k.
                for x in points:
                    if len(block) >= k:
                        break
                    if x not in in_block:
                        block.append(x)
                        in_block.add(x)
                break
            block.append(best_x)

        block.sort()
        blocks.append(block)

        # Remove the subsets this block covers.
        for c in itertools.combinations(block, t):
            m = 0
            for y in c:
                m |= 1 << y
            uncovered.discard(m)

        # Dominance removal: drop any existing block that is a subset of the
        # new block (its t-subsets are all covered by the new block).
        pm = 0
        for x in block:
            pm |= 1 << x
        new_blocks = []
        for b in blocks[:-1]:
            bm = 0
            for x in b:
                bm |= 1 << x
            if (bm | pm) == pm:
                # b is a subset of block -> redundant
                continue
            new_blocks.append(b)
        new_blocks.append(block)
        blocks = new_blocks

    return blocks
