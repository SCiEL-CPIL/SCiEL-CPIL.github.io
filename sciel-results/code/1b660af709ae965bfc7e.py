"""Covering design C(19,7,4) - optimized greedy + improvement."""
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

_DEADLINE = 145.0  # safety margin before 150s kill


def _build_index(v, t):
    """Map each t-subset to an integer index for fast bitset ops."""
    idx = {}
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        i += 1
    return idx, i  # index dict, total count


def _block_bitset(block, v, t, idx):
    """Return a numpy bool array of length C(v,t) marking covered t-subsets."""
    n = len(idx)
    mask = np.zeros(n, dtype=bool)
    for comb in combinations(block, t):
        mask[idx[comb]] = True
    return mask


def _greedy_construct(v, k, t, idx, total, deadline):
    """Fast greedy using precomputed pair-coverage counts."""
    n = total
    uncovered = np.ones(n, dtype=bool)
    blocks = []
    start = time.time()

    while uncovered.any():
        if time.time() > deadline:
            break

        # Find first uncovered t-subset
        first_idx = np.argmax(uncovered)
        first_tsub = next(c for c, i in idx.items() if i == first_idx)
        block = list(first_tsub)

        # Greedily add points: pick x maximizing new coverage
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many uncovered t-subsets would be covered
                gain = 0
                for rest in combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    if idx.get(tsub, -1) < n and uncovered[idx[tsub]]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)
            if time.time() > deadline:
                break

        block.sort()
        blocks.append(block)
        # Update uncovered
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                uncovered[i] = False

    return blocks


def _improve_blocks(blocks, v, k, t, idx, deadline):
    """Try to remove redundant blocks."""
    n = len(idx)
    # Build coverage count per t-subset
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    blocks = list(blocks)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Check if removing this block leaves any t-subset uncovered
            removable = True
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0 and coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(comb, -1)
                    if ci >= 0:
                        coverage[ci] -= 1
                blocks.pop(i)
                improved = True
                break  # restart scan from beginning

    return blocks


def _random_removal_pass(blocks, v, k, t, idx, deadline, n_removals=5000):
    """Randomized attempt to find removable blocks (handles ordering issues)."""
    n = len(idx)
    blocks = list(blocks)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            i = idx.get(comb, -1)
            if i >= 0:
                coverage[i] += 1

    rng = random.Random(42)
    for _ in range(n_removals):
        if time.time() > deadline:
            break
        if not blocks:
            break
        i = rng.randrange(len(blocks))
        block = blocks[i]
        removable = True
        for comb in combinations(block, t):
            ci = idx.get(comb, -1)
            if ci >= 0 and coverage[ci] <= 1:
                removable = False
                break
        if removable:
            for comb in combinations(block, t):
                ci = idx.get(comb, -1)
                if ci >= 0:
                    coverage[ci] -= 1
            blocks.pop(i)

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index
    idx, total = _build_index(v, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, deadline)

    # Deterministic improvement pass
    if time.time() < deadline - 5:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)

    # Randomized removal pass
    if time.time() < deadline - 3:
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline)

    # Final safety: ensure we return something valid if we ran out of time
    # (greedy always produces a valid covering if it completes)
    return blocks
