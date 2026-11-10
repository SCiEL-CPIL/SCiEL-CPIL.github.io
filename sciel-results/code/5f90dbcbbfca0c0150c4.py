"""Combined covering design C(19,8,5): bitmask-accelerated greedy with bottleneck heuristic,
randomized restarts, and redundancy pruning."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def _precompute(v, k, t):
    """Precompute t-subsets as integer bitmasks and lookup structures."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    # bitmask for each t-subset
    tsub_masks = np.zeros(n_tsub, dtype=np.int64)
    tsub_to_idx = {}
    for i, ts in enumerate(all_tsubsets):
        m = 0
        for x in ts:
            m |= (1 << x)
        tsub_masks[i] = m
        tsub_to_idx[ts] = i
    return all_tsubsets, tsub_masks, tsub_to_idx, n_tsub


def _block_tsub_indices(block, v, t, tsub_to_idx):
    """Return indices of t-subsets contained in block."""
    return [tsub_to_idx[ts] for ts in itertools.combinations(block, t) if ts in tsub_to_idx]


def _fast_greedy_bottleneck(v, k, t, seed, max_time, randomize=False, use_bottleneck=True):
    """
    Greedy covering using bitmask operations.
    Block selection heuristic:
      - If use_bottleneck: pick the uncovered t-subset with the fewest candidate
        extensions (bottleneck), then build a block around it.
      - Otherwise: pick the block that covers the most uncovered t-subsets.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time

    all_tsubsets, tsub_masks, tsub_to_idx, n_tsub = _precompute(v, k, t)

    covered = np.zeros(n_tsub, dtype=bool)
    num_uncovered = n_tsub
    blocks = []
    seen_blocks = set()

    while num_uncovered > 0:
        if time.time() > deadline:
            break

        uncovered_mask = ~covered
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break

        # --- Build candidate block ---
        # Choose a focus t-subset
        if use_bottleneck:
            # Bottleneck: among uncovered t-subsets, prefer those that are "hard" to cover.
            # Approximate hardness by how many other uncovered t-subsets share >= t-1 points
            # with it (i.e., can be covered by a block extending it).
            # For speed, sample if many.
            if randomize and len(uncovered_indices) > 50:
                sample_idx = rng.sample(list(uncovered_indices), min(50, len(uncovered_indices)))
            else:
                sample_idx = list(uncovered_indices[:50])

            best_focus = None
            best_hardness = -1
            for idx in sample_idx:
                focus = all_tsubsets[idx]
                # Count uncovered t-subsets sharing at least t-1 elements with focus
                # A t-subset shares t-1 elements with focus iff their bitmask intersection has t-1 bits
                fmask = tsub_masks[idx]
                # Intersect with all uncovered masks
                # For each uncovered j, popcount(fmask & mask_j) >= t-1
                inter = np.bitwise_and(fmask, tsub_masks[uncovered_indices])
                # Count bits
                hardness = np.sum([bin(x).count('1') for x in inter])
                if hardness > best_hardness:
                    best_hardness = hardness
                    best_focus = idx
            if best_focus is None:
                best_focus = sample_idx[0]
        else:
            # Just pick first uncovered or random
            if randomize and len(uncovered_indices) > 1:
                best_focus = rng.choice(list(uncovered_indices))
            else:
                best_focus = uncovered_indices[0]

        focus = all_tsubsets[best_focus]
        block_set = set(focus)

        # Greedily extend block: add point that covers most uncovered t-subsets
        while len(block_set) < k:
            if time.time() > deadline:
                break
            best_point = None
            best_local_gain = -1

            for x in range(v):
                if x in block_set:
                    continue
                # Count uncovered t-subsets covered by adding x to block_set
                # A t-subset is covered if it contains x and t-1 elements in block_set
                gain = 0
                for rest in itertools.combinations(block_set, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    i = tsub_to_idx.get(ts, -1)
                    if i != -1 and not covered[i]:
                        gain += 1
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_point = x
                elif gain == best_local_gain and best_point is not None and randomize and rng.random() < 0.3:
                    best_point = x

            if best_point is None:
                break
            block_set.add(best_point)

        block = tuple(sorted(block_set))
        if block in seen_blocks:
            # Try a different focus
            if randomize:
                alt = rng.choice(list(uncovered_indices))
                block_set = set(all_tsubsets[alt])
                while len(block_set) < k:
                    if time.time() > deadline:
                        break
                    best_point = None
                    best_local_gain = -1
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in itertools.combinations(block_set, t - 1):
                            ts = tuple(sorted(rest + (x,)))
                            i = tsub_to_idx.get(ts, -1)
                            if i != -1 and not covered[i]:
                                gain += 1
                        if gain > best_local_gain:
                            best_local_gain = gain
                            best_point = x
                    if best_point is None:
                        break
                    block_set.add(best_point)
                block = tuple(sorted(block_set))
                if block in seen_blocks:
                    continue
            else:
                continue

        seen_blocks.add(block)
        blocks.append(list(block))

        # Update covered
        for ts in itertools.combinations(block, t):
            i = tsub_to_idx.get(ts, -1)
            if i != -1 and not covered[i]:
                covered[i] = True
                num_uncovered -= 1

    return blocks


def _prune_blocks(blocks, v, k, t, max_time=10.0):
    """Remove redundant blocks: a block is redundant if all its t-subsets are covered by other blocks."""
    start_time = time.time()
    deadline = start_time + max_time

    all_tsubsets, tsub_masks, tsub_to_idx, n_tsub = _precompute(v, k, t)

    # Count coverage: for each t-subset, how many blocks cover it
    tsub_count = np.zeros(n_tsub, dtype=np.int32)
    block_tsubs = []  # for each block, list of t-subset indices it covers
    for block in blocks:
        indices = _block_tsub_indices(block, v, t, tsub_to_idx)
        block_tsubs.append(indices)
        for i in indices:
            tsub_count[i] += 1

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            redundant = True
            for idx in block_tsubs[i]:
                if tsub_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for idx in block_tsubs[i]:
                    tsub_count[idx] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                improved = True

    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    total_start = time.time()
    total_deadline = total_start + 140.0  # Safety margin before 150s kill

    best_blocks = None
    best_len = float('inf')

    # Multiple strategies with different seeds
    seeds = [1, 42, 123, 777, 2024, 999, 555, 314, 271, 1618]

    for i, seed in enumerate(seeds):
        if time.time() > total_deadline:
            break
        remaining = total_deadline - time.time()
        if remaining < 3.0:
            break

        # Alternate between bottleneck and pure greedy
        use_bn = (i % 2 == 0)
        randomize = (i > 0)
        alloc = min(remaining * 0.45, 25.0) if i < 3 else min(remaining * 0.3, 15.0)

        blocks = _fast_greedy_bottleneck(v, k, t, seed=seed, max_time=alloc,
                                         randomize=randomize, use_bottleneck=use_bn)

        if blocks and len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        return []

    # Prune redundant blocks
    if time.time() < total_deadline - 5.0:
        refine_time = min(total_deadline - time.time() - 5.0, 15.0)
        best_blocks = _prune_blocks(best_blocks, v, k, t, max_time=refine_time)

    return best_blocks
# EVOLVE-BLOCK-END
