"""Covering design C(19,7,4) solver: greedy + local search refinement."""
# EVOLVE-BLOCK-START
import time
import itertools
import random
from math import comb

# Global deadline guard
_DEADLINE = None
_TIME_LIMIT = 115.0  # seconds, safety margin before 120s kill


def _time_remaining():
    if _DEADLINE is None:
        return _TIME_LIMIT
    return max(0.0, _DEADLINE - time.time())


def construct(v, k, t):
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index for fast lookup
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Bitset representation: use Python ints as bitsets
    # Each block covers a set of t-subsets; represent as a bitmask
    # n_tsubs can be up to C(19,4)=3876, fits in a Python int

    # Precompute for each k-subset (block) the bitmask of t-subsets it covers
    # Actually, precompute for each (k-subset) -> bitmask
    # But there are C(19,7)=50388 blocks; each has C(7,4)=35 t-subsets
    # Precompute bitmask for each block
    block_bitmasks = {}
    for blk in itertools.combinations(range(v), k):
        mask = 0
        for ts in itertools.combinations(blk, t):
            mask |= (1 << tsub_index[ts])
        block_bitmasks[blk] = mask

    # Greedy construction
    uncovered_mask = (1 << n_tsubs) - 1
    blocks = []

    # For greedy: pick block that covers most uncovered t-subsets
    # To speed up, we can use a heuristic: start from an uncovered t-subset,
    # then greedily add points to maximize coverage.

    while uncovered_mask:
        if _time_remaining() < 5.0:
            break

        # Find first uncovered t-subset to anchor the block
        # Find the lowest set bit
        # Get an uncovered t-subset
        # Use a random uncovered t-subset for diversity
        # First, find any uncovered t-subset
        # Count bits: use bit_length trick
        # Find index of a set bit
        # Use: idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        anchor_tsub = all_tsubs[idx]

        # Greedily build a block containing this t-subset
        block = list(anchor_tsub)
        block_set = set(block)

        # Try to extend: add point that maximizes newly covered t-subsets
        while len(block) < k:
            if _time_remaining() < 5.0:
                break
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets in (block + {x})
                # that contain at least one element from block
                # A t-subset covered by block∪{x} but not by block must contain x
                # and (t-1) elements from block
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    bit = 1 << tsub_index[ts]
                    if uncovered_mask & bit:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            # Fill remaining with arbitrary points
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)

        block.sort()
        blocks.append(block)
        uncovered_mask &= ~block_bitmasks[tuple(block)]

    # Local search refinement: try to swap points in blocks to reduce count
    # or try to remove redundant blocks
    # Strategy: try to find a block that is redundant (all its t-subsets covered by others)
    # and remove it. Repeat.

    def _try_remove_redundant(blocks):
        """Try to remove blocks that are fully covered by others."""
        changed = True
        while changed and _time_remaining() > 2.0:
            changed = False
            n = len(blocks)
            for i in range(n):
                if _time_remaining() < 2.0:
                    break
                # Check if block i is redundant
                # Compute union of all other blocks' coverage
                # For efficiency, check if every t-subset of block i is in some other block
                bi = blocks[i]
                redundant = True
                for ts in itertools.combinations(bi, t):
                    found = False
                    for j in range(n):
                        if j == i:
                            continue
                        if all(x in blocks[j] for x in ts):
                            found = True
                            break
                    if not found:
                        redundant = False
                        break
                if redundant:
                    blocks.pop(i)
                    changed = True
                    break  # restart scan
        return blocks

    def _try_swap_improvement(blocks):
        """Try to improve by swapping a point in one block with a point from another,
        or by replacing a block with a different one that covers same or more."""
        improved = True
        while improved and _time_remaining() > 2.0:
            improved = False
            n = len(blocks)
            for i in range(n):
                if _time_remaining() < 2.0:
                    break
                bi = blocks[i]
                for pos in range(k):
                    if _time_remaining() < 2.0:
                        break
                    orig = bi[pos]
                    for x in range(v):
                        if x == orig:
                            continue
                        # Try replacing orig with x
                        new_bi = bi[:pos] + [x] + bi[pos+1:]
                        new_bi.sort()
                        new_mask = block_bitmasks.get(tuple(new_bi), 0)
                        # Check: does new block cover at least what old block uniquely covers?
                        # Old block's unique coverage = bits in old_mask not in union of others
                        # This is expensive; simpler: just try and see if total coverage improves
                        # Actually, we want to minimize block count, so swapping within same count
                        # only helps if it makes another block redundant.
                        # Simpler heuristic: swap if new block covers more uncovered t-subsets
                        # relative to current state. But we don't track uncovered easily here.
                        # Let's skip this for now and just do removal.
                        pass
        return blocks

    # Try removing redundant blocks
    blocks = _try_remove_redundant(blocks)

    # Try a second pass of greedy on remaining uncovered, in case removal exposed new options
    # Actually, after removal, re-check coverage
    uncovered_mask = (1 << n_tsubs) - 1
    for b in blocks:
        uncovered_mask &= ~block_bitmasks[tuple(b)]

    while uncovered_mask and _time_remaining() > 3.0:
        idx = (uncovered_mask & -uncovered_mask).bit_length() - 1
        anchor_tsub = all_tsubs[idx]
        block = list(anchor_tsub)
        block_set = set(block)
        while len(block) < k:
            if _time_remaining() < 3.0:
                break
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    bit = 1 << tsub_index[ts]
                    if uncovered_mask & bit:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)
        if len(block) < k:
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
        block.sort()
        blocks.append(block)
        uncovered_mask &= ~block_bitmasks[tuple(block)]

    # Final redundancy removal
    blocks = _try_remove_redundant(blocks)

    return blocks
# EVOLVE-BLOCK-END
