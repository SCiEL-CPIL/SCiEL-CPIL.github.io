"""Improved covering design with optimized greedy and redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    all_t_subs = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subs)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    uncovered = set(range(n_t))
    blocks = []

    def time_ok():
        return time.time() < deadline

    # Phase 1: Greedy
    while uncovered and time_ok():
        first_idx = min(uncovered)
        block = list(all_t_subs[first_idx])
        block_set = set(block)

        while len(block) < k and time_ok():
            best_x = -1
            best_gain = -1
            t_minus_1 = list(itertools.combinations(block, t - 1))

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in t_minus_1:
                    cand = tuple(sorted(rest + (x,)))
                    idx = t_sub_to_idx.get(cand)
                    if idx is not None and idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx.get(t_sub)
            if idx is not None:
                uncovered.discard(idx)

    if uncovered:
        return blocks

    # Phase 2: Vectorized redundancy removal using coverage counts
    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks

    # Build index: for each block, precompute the list of t-subset indices it covers
    block_t_indices = []
    for b in blocks:
        idxs = []
        for t_sub in itertools.combinations(b, t):
            idx = t_sub_to_idx.get(t_sub)
            if idx is not None:
                idxs.append(idx)
        block_t_indices.append(np.array(idxs, dtype=np.int32))

    # Initialize coverage count array
    cov = np.zeros(n_t, dtype=np.int32)
    for idxs in block_t_indices:
        if len(idxs) > 0:
            np.add.at(cov, idxs, 1)

    # Iteratively remove blocks that have no uniquely covered t-subset
    removed = np.zeros(n_blocks, dtype=bool)
    changed = True
    while changed and time_ok():
        changed = False
        for i in range(n_blocks):
            if removed[i]:
                continue
            idxs = block_t_indices[i]
            if len(idxs) == 0:
                # Block covers no t-subsets, safe to remove
                removed[i] = True
                changed = True
                continue
            # Check if any t-subset in this block has coverage count == 1
            # (meaning this block is the only one covering it)
            counts = cov[idxs]
            if np.any(counts == 1):
                continue  # Block is needed
            # All t-subsets in this block are covered by other blocks
            # Remove this block
            removed[i] = True
            changed = True
            # Update coverage counts
            np.add.at(cov, idxs, -1)

    # Build result list
    result = [blocks[i] for i in range(n_blocks) if not removed[i]]
    return result
# EVOLVE-BLOCK-END
