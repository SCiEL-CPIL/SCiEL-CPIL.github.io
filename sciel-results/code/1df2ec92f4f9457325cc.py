"""Covering design C(v,k,t) via randomized greedy with local search and time guard."""
import itertools
import random
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START


def _all_tsubsets(v, t):
    """Generate all t-subsets of range(v) as frozensets."""
    return [frozenset(c) for c in itertools.combinations(range(v), t)]


def _count_tsubsets_in_block(block, t):
    """Count all t-subsets of a block."""
    return [frozenset(c) for c in itertools.combinations(block, t)]


def _greedy_cover(v, k, t, seed=0, deadline=None):
    """
    Randomized greedy covering.
    Returns list of blocks.
    """
    rng = random.Random(seed)
    all_tsubs = _all_tsubsets(v, t)
    total = len(all_tsubs)
    # coverage count per t-subset
    covered = [0] * total
    # map frozenset -> index
    tsub_index = {fs: i for i, fs in enumerate(all_tsubs)}
    uncovered_count = total

    blocks = []
    points = list(range(v))

    while uncovered_count > 0:
        if deadline is not None and time.time() > deadline:
            break

        # Pick the most constrained uncovered t-subset (one with lowest coverage, i.e., 0)
        # To add randomness, pick randomly among uncovered
        uncovered_list = [i for i in range(total) if covered[i] == 0]
        if not uncovered_list:
            break
        # Pick a random uncovered t-subset to start the block
        start_sub = all_tsubsets[rng.choice(uncovered_list)]
        block = list(start_sub)

        # Greedily add points to maximize new coverage
        while len(block) < k:
            if deadline is not None and time.time() > deadline:
                break
            candidates = [x for x in points if x not in block]
            if not candidates:
                break

            # For each candidate, count how many new t-subsets it covers
            # A new t-subset is one that contains x and (t-1) elements from block
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    fs = frozenset(rest) | {x}
                    idx = tsub_index[fs]
                    if covered[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # No improvement; pick randomly to break ties
                best_x = rng.choice(candidates)

            block.append(best_x)

        if len(block) < k:
            # Fill remaining with random points
            remaining = [x for x in points if x not in block]
            rng.shuffle(remaining)
            block.extend(remaining[:k - len(block)])

        block.sort()

        # Check if this block is already in our list
        block_tuple = tuple(block)
        if block_tuple not in set(b for b in blocks):
            blocks.append(block)
            # Update coverage
            for tsub in _count_tsubsets_in_block(block, t):
                idx = tsub_index[tsub]
                if covered[idx] == 0:
                    uncovered_count -= 1
                covered[idx] += 1

    return blocks


def _local_search(blocks, v, k, t, deadline=None, seed=0):
    """
    Try to improve the covering by removing blocks that are redundant
    (all their t-subsets are still covered by other blocks).
    Also try swapping: replace a block with one that covers more uncovered t-subsets.
    """
    rng = random.Random(seed)
    if not blocks:
        return blocks

    all_tsubs = _all_tsubsets(v, t)
    total = len(all_tsubs)
    tsub_index = {fs: i for i, fs in enumerate(all_tsubs)}

    # Compute coverage counts
    covered = [0] * total
    for block in blocks:
        for tsub in _count_tsubsets_in_block(block, t):
            idx = tsub_index[tsub]
            covered[idx] += 1

    # Try to remove redundant blocks
    changed = True
    while changed:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if deadline is not None and time.time() > deadline:
                break
            # Check if removing block i still covers everything
            removable = True
            for tsub in _count_tsubsets_in_block(blocks[i], t):
                idx = tsub_index[tsub]
                if covered[idx] <= 1:
                    removable = False
                    break
            if removable:
                for tsub in _count_tsubsets_in_block(blocks[i], t):
                    idx = tsub_index[tsub]
                    covered[idx] -= 1
                blocks.pop(i)
                changed = True

    return blocks


def _try_swap_improve(blocks, v, k, t, deadline=None, seed=0):
    """
    Try to replace some blocks to reduce total count.
    For each block, see if we can find a new block that covers all its uniquely-covered t-subsets.
    This is complex; instead, try a simple approach:
    - Identify blocks that cover unique t-subsets
    - For blocks that don't cover unique t-subsets, they are redundant (handled above)
    - For blocks that do cover unique t-subsets, try to find a replacement
    """
    # This is a simplified version; the main improvement comes from greedy + removal
    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    # Leave 15 seconds safety margin
    deadline = start_time + 105.0

    # Multiple randomized greedy restarts
    best_blocks = None
    best_len = float('inf')

    # First, do a deterministic greedy (seed=0)
    blocks = _greedy_cover(v, k, t, seed=0, deadline=deadline)
    if blocks:
        blocks = _local_search(blocks, v, k, t, deadline=deadline, seed=0)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks[:]

    # Try multiple random restarts
    for seed in range(1, 200):
        if time.time() > deadline:
            break
        blocks = _greedy_cover(v, k, t, seed=seed, deadline=deadline)
        if blocks:
            blocks = _local_search(blocks, v, k, t, deadline=deadline, seed=seed)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks[:]

    if best_blocks is None:
        # Fallback: should not happen, but just in case
        return []

    return best_blocks


# EVOLVE-BLOCK-END
