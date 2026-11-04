"""
Covering design C(v, k, t) via randomized greedy with bitmask acceleration.
"""

import itertools
import random
import time

DEADLINE_SECONDS = 140.0


def construct(v, k, t):
    start_wall = time.time()
    rng = random.Random(42)

    # Precompute all t-subsets and their masks
    subset_list = []
    for combo in itertools.combinations(range(v), t):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        subset_list.append((combo, mask))
    n_subsets = len(subset_list)

    # mask -> index
    mask_to_idx = {mask: i for i, (combo, mask) in enumerate(subset_list)}

    # For each point x, list of (t_subset_mask, rest_mask) where rest_mask is
    # the (t-1)-subset of points (excluding x) in that t-subset.
    masks_containing = [[] for _ in range(v)]
    for combo, mask in subset_list:
        for x in combo:
            rest = [y for y in combo if y != x]
            rest_mask = 0
            for y in rest:
                rest_mask |= (1 << y)
            masks_containing[x].append((mask, rest_mask))

    def fast_greedy(point_order, time_limit):
        uncovered = bytearray([1] * n_subsets)
        blocks = []
        local_start = time.time()

        while True:
            if time.time() - local_start > time_limit:
                break

            # Find uncovered indices
            unc_list = [i for i in range(n_subsets) if uncovered[i]]
            if not unc_list:
                break

            chosen = rng.choice(unc_list)
            combo, mask = subset_list[chosen]

            block_set = set(combo)
            block_mask = 0
            for x in combo:
                block_mask |= (1 << x)

            while len(block_set) < k:
                if time.time() - local_start > time_limit:
                    break

                best_x = -1
                best_gain = -1
                for x in point_order:
                    if x in block_set:
                        continue
                    gain = 0
                    for (m, rest_mask) in masks_containing[x]:
                        if rest_mask & ~block_mask:
                            continue
                        idx = mask_to_idx.get(m)
                        if idx is not None and uncovered[idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1 or best_gain == 0:
                    unused = [x for x in point_order if x not in block_set]
                    if not unused:
                        break
                    best_x = rng.choice(unused)

                block_set.add(best_x)
                block_mask |= (1 << best_x)

            block_list = sorted(block_set)
            blocks.append(block_list)

            for combo_t in itertools.combinations(block_list, t):
                m = 0
                for y in combo_t:
                    m |= (1 << y)
                idx = mask_to_idx.get(m)
                if idx is not None:
                    uncovered[idx] = 0

        return blocks

    best_blocks = None
    best_count = float('inf')

    # Initial pass
    natural_order = list(range(v))
    remaining = DEADLINE_SECONDS - (time.time() - start_wall) - 5
    if remaining > 5:
        blocks = fast_greedy(natural_order, remaining)
        if blocks:
            best_blocks = blocks
            best_count = len(blocks)

    # Randomized restarts
    while time.time() - start_wall < DEADLINE_SECONDS - 10:
        perm = list(range(v))
        rng.shuffle(perm)
        blocks = fast_greedy(perm, DEADLINE_SECONDS - (time.time() - start_wall) - 5)
        if blocks and len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)

    if best_blocks is None:
        return []
    return best_blocks
