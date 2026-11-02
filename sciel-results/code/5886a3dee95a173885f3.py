"""Covering design C(19,8,5): bitset-accelerated randomized greedy with redundancy removal."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter


def _all_t_subsets(v, t):
    """Return list of all t-subsets as sorted tuples."""
    return list(itertools.combinations(range(v), t))


def _subset_to_idx_map(t_subs):
    """Map each t-subset tuple to its index."""
    return {s: i for i, s in enumerate(t_subs)}


def _block_mask(block, t, idx_map):
    """Return set of indices of t-subsets covered by block."""
    return {idx_map[tuple(sorted(c))] for c in itertools.combinations(block, t)}


def _greedy_build(v, k, t, t_subs, idx_map, seed=None, deadline=None, safety_margin=5.0):
    """Randomized greedy: repeatedly pick uncovered t-subset, build block maximizing coverage."""
    rng = random.Random(seed)
    n = len(t_subs)
    # uncovered as a set of indices
    uncovered = set(range(n))
    blocks = []
    start = time.time()

    while uncovered:
        if deadline and (time.time() - start) > deadline - safety_margin:
            break

        # Pick a random uncovered t-subset to start a block
        start_idx = rng.choice(list(uncovered))
        start_subset = t_subs[start_idx]
        block = list(start_subset)

        while len(block) < k:
            if deadline and (time.time() - start) > deadline - safety_margin:
                break

            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: number of uncovered t-subsets that would be covered
                # if we add x to block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    idx = idx_map[candidate]
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is not None:
                block.append(best_x)

        block.sort()
        blocks.append(block)
        # Remove covered t-subsets
        mask = _block_mask(block, t, idx_map)
        uncovered -= mask

    return blocks


def _remove_redundant_blocks(blocks, t, t_subs, idx_map, deadline=None, safety_margin=5.0):
    """Remove blocks that are redundant (their t-subsets are all covered by other blocks)."""
    if not blocks:
        return blocks

    n_blocks = len(blocks)
    # For each block, compute the set of t-subset indices it covers
    block_masks = [_block_mask(b, t, idx_map) for b in blocks]

    # Count how many blocks cover each t-subset
    cover_count = Counter()
    for mask in block_masks:
        for idx in mask:
            cover_count[idx] += 1

    result = []
    start = time.time()
    for i, block in enumerate(blocks):
        if deadline and (time.time() - start) > deadline - safety_margin:
            result.append(block)
            continue

        # Check if this block is redundant: all its t-subsets are covered by other blocks
        redundant = True
        for idx in block_masks[i]:
            if cover_count[idx] == 1:
                redundant = False
                break
        if not redundant:
            result.append(block)
            # Decrement cover counts for this block's t-subsets
            for idx in block_masks[i]:
                cover_count[idx] -= 1
        # If redundant, skip it (don't decrement counts)

    return result


def _verify_covering(blocks, v, t):
    """Verify every t-subset is covered by at least one block."""
    covered = set()
    for block in blocks:
        for c in itertools.combinations(block, t):
            covered.add(tuple(sorted(c)))
    all_subs = set(itertools.combinations(range(v), t))
    return all_subs == covered


def construct(v, k, t):
    """Construct a covering design: every t-subset of {0..v-1} is in some k-block."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx_map(t_subs)

    best_blocks = None
    best_count = float('inf')

    # Try multiple random seeds with greedy
    max_seeds = 10
    for seed in range(max_seeds):
        if time.time() > deadline - 20.0:
            break

        blocks = _greedy_build(v, k, t, t_subs, idx_map, seed=seed,
                                deadline=deadline, safety_margin=10.0)
        if not blocks:
            continue

        # Remove redundant blocks
        blocks = _remove_redundant_blocks(blocks, t, t_subs, idx_map,
                                           deadline=deadline, safety_margin=10.0)

        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

        # Early stop if we hit theoretical lower bound
        import math
        lower_bound = math.ceil(len(t_subs) / math.comb(k, t))
        if best_count <= lower_bound:
            break

    if best_blocks is None:
        # Fallback: simple greedy without redundancy removal
        best_blocks = _greedy_build(v, k, t, t_subs, idx_map, seed=0,
                                     deadline=deadline, safety_margin=5.0)

    # Final verification
    if not _verify_covering(best_blocks, v, t):
        # Should not happen, but if it does, extend with more greedy
        uncovered = set(itertools.combinations(range(v), t))
        for block in best_blocks:
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))
        while uncovered:
            if time.time() > deadline - 5.0:
                break
            start_subset = min(uncovered)
            block = list(start_subset)
            while len(block) < k:
                best_x = None
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        candidate = tuple(sorted(rest + (x,)))
                        if candidate in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    block.append(best_x)
            block.sort()
            best_blocks.append(block)
            for c in itertools.combinations(block, t):
                uncovered.discard(tuple(sorted(c)))

    return best_blocks
# EVOLVE-BLOCK-END
