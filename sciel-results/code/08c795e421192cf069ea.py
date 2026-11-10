"""Covering design C(19,7,4) - fast bitset greedy + local search."""
import time
import random
import numpy as np
from itertools import combinations

_DEADLINE = 148.0


def _build_index(v, t):
    """Map each t-subset to an integer index and build reverse list."""
    idx = {}
    rev = []
    i = 0
    for comb in combinations(range(v), t):
        idx[comb] = i
        rev.append(comb)
        i += 1
    return idx, rev, i


def _block_bitset(block, idx, t):
    """Return frozenset of t-subset indices covered by a block."""
    result = set()
    for comb in combinations(block, t):
        ci = idx.get(tuple(comb))
        if ci is not None:
            result.add(ci)
    return frozenset(result)


def _greedy_construct(v, k, t, idx, rev, total, deadline):
    """Fast greedy: pick block covering most uncovered t-subsets."""
    uncovered = set(range(total))
    blocks = []
    start = time.time()

    # Precompute for each point, the t-subsets containing it
    point_to_tsubs = [[] for _ in range(v)]
    for i, comb in enumerate(rev):
        for pt in comb:
            point_to_tsubs[pt].append(i)

    while uncovered:
        if time.time() > deadline:
            break

        # Pick an uncovered t-subset to anchor
        anchor = min(uncovered)
        anchor_comb = rev[anchor]
        block = list(anchor_comb)
        block_set = set(block)

        # Greedily add points that maximize new coverage
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count new uncovered t-subsets that would be covered
                gain = 0
                for rest in combinations(block, t - 1):
                    tsub = tuple(sorted(rest + (x,)))
                    ci = idx.get(tsub)
                    if ci is not None and ci in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x == -1 or best_gain == 0:
                break
            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            # Fill with random points
            remaining = [x for x in range(v) if x not in block_set]
            random.shuffle(remaining)
            block.extend(remaining[:k - len(block)])

        block.sort()
        blocks.append(block)

        # Update uncovered
        for comb in combinations(block, t):
            ci = idx.get(tuple(comb))
            if ci is not None:
                uncovered.discard(ci)

    return blocks


def _compute_coverage(blocks, v, t, idx):
    """Compute coverage count for each t-subset."""
    n = len(idx)
    coverage = np.zeros(n, dtype=np.int32)
    for block in blocks:
        for comb in combinations(block, t):
            ci = idx.get(tuple(comb))
            if ci is not None:
                coverage[ci] += 1
    return coverage


def _remove_redundant(blocks, v, t, idx, deadline):
    """Remove blocks that don't uniquely cover any t-subset."""
    n = len(idx)
    coverage = _compute_coverage(blocks, v, t, idx)
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
            removable = True
            for comb in combinations(block, t):
                ci = idx.get(tuple(comb))
                if ci is not None and coverage[ci] <= 1:
                    removable = False
                    break
            if removable:
                for comb in combinations(block, t):
                    ci = idx.get(tuple(comb))
                    if ci is not None:
                        coverage[ci] -= 1
                blocks.pop(i)
                improved = True
                break
    return blocks, coverage


def _local_search(blocks, v, k, t, idx, deadline, rng):
    """Local search: try removing blocks and swapping."""
    n = len(idx)
    blocks = list(blocks)
    coverage = _compute_coverage(blocks, v, t, idx)
    best_blocks = list(blocks)
    best_len = len(blocks)

    # Precompute t-subsets for each block
    block_tsubs = []
    for block in blocks:
        tsubs = []
        for comb in combinations(block, t):
            ci = idx.get(tuple(comb))
            if ci is not None:
                tsubs.append(ci)
        block_tsubs.append(tsubs)

    start_time = time.time()
    max_iters = 200000

    for it in range(max_iters):
        if time.time() > deadline:
            break

        move_type = rng.random()
        if move_type < 0.6:
            # Try to remove a block
            if not blocks:
                break
            i = rng.randrange(len(blocks))
            tsubs = block_tsubs[i]
            can_remove = all(coverage[ci] > 1 for ci in tsubs)
            if can_remove:
                for ci in tsubs:
                    coverage[ci] -= 1
                blocks.pop(i)
                block_tsubs.pop(i)
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
        else:
            # Try to swap a block
            if len(blocks) < 2:
                continue
            i = rng.randrange(len(blocks))
            old_tsubs = block_tsubs[i]
            can_remove = all(coverage[ci] > 1 for ci in old_tsubs)
            if not can_remove:
                continue

            # Generate a new block
            new_block = sorted(rng.sample(range(v), k))
            new_tsubs = []
            for comb in combinations(new_block, t):
                ci = idx.get(tuple(comb))
                if ci is not None:
                    new_tsubs.append(ci)

            # Remove old, add new
            for ci in old_tsubs:
                coverage[ci] -= 1
            for ci in new_tsubs:
                coverage[ci] += 1

            # Check validity
            valid = True
            for ci in range(n):
                if coverage[ci] == 0:
                    valid = False
                    break

            if not valid:
                # Undo
                for ci in new_tsubs:
                    coverage[ci] -= 1
                for ci in old_tsubs:
                    coverage[ci] += 1
            else:
                blocks[i] = new_block
                block_tsubs[i] = new_tsubs
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)

    return best_blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index
    idx, rev, total = _build_index(v, t)

    # Greedy construction
    blocks = _greedy_construct(v, k, t, idx, rev, total, deadline)

    # Remove redundant blocks
    if time.time() < deadline - 15:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    # Local search
    if time.time() < deadline - 10:
        rng = random.Random(42)
        blocks = _local_search(blocks, v, k, t, idx, deadline, rng)

    # Final removal pass
    if time.time() < deadline - 3:
        blocks, _ = _remove_redundant(blocks, v, t, idx, deadline)

    return blocks
