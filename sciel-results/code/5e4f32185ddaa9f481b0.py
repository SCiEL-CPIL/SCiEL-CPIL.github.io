"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _greedy_cover(v, k, t, rng=None, deadline=None):
    """Greedy set-cover: open a block on the first uncovered t-subset, then add
    the point that covers the most uncovered t-subsets until the block has k points."""
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = float('inf')
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset as seed (randomization helps)
        seed = rng.choice(list(uncovered))
        block = list(seed)
        block_set = set(block)
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            candidates = [x for x in range(v) if x not in block]
            block.append(max(candidates, key=gain))
        block.sort()
        blocks.append(block)
        block_set = set(block)
        for ts in itertools.combinations(block, t):
            uncovered.discard(tuple(sorted(ts)))
    return blocks


def _build_coverage(blocks, v, t):
    """Map t-subset -> set of block indices that contain it."""
    coverage = {}
    for bi, block in enumerate(blocks):
        for ts in itertools.combinations(block, t):
            ts = tuple(sorted(ts))
            if ts not in coverage:
                coverage[ts] = set()
            coverage[ts].add(bi)
    return coverage


def _remove_block(blocks, bi, coverage, t):
    """Remove block bi, update coverage, re-index. Returns newly uncovered t-subsets."""
    block = blocks[bi]
    newly_uncovered = []
    for ts in itertools.combinations(block, t):
        ts = tuple(sorted(ts))
        cov = coverage.get(ts, set())
        if len(cov) == 1 and bi in cov:
            newly_uncovered.append(ts)
        coverage[ts].discard(bi)
        if not coverage[ts]:
            del coverage[ts]
    blocks.pop(bi)
    # Re-index: all blocks after bi shift down by 1
    for ts in list(coverage.keys()):
        new_set = set()
        for idx in coverage[ts]:
            if idx < bi:
                new_set.add(idx)
            else:
                new_set.add(idx - 1)
        coverage[ts] = new_set
    return newly_uncovered


def _add_block(blocks, block, coverage, t):
    """Add a block, update coverage. Returns index of new block."""
    bi = len(blocks)
    blocks.append(block)
    for ts in itertools.combinations(block, t):
        ts = tuple(sorted(ts))
        if ts not in coverage:
            coverage[ts] = set()
        coverage[ts].add(bi)
    return bi


def _greedy_fill(block, uncovered, v, k, t):
    """Fill block up to size k by greedily adding points that cover most uncovered t-subsets."""
    block_set = set(block)
    while len(block) < k:
        best_x = None
        best_gain = -1
        for x in range(v):
            if x in block_set:
                continue
            gain = sum(1 for rest in itertools.combinations(block, t - 1)
                       if tuple(sorted(rest + (x,))) in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None or best_gain == 0:
            break
        block.append(best_x)
        block_set.add(best_x)
    return block


def _local_search(blocks, v, k, t, deadline):
    """Try to reduce block count via local search:
    1. Remove redundant blocks (those with no exclusive t-subsets).
    2. Try swapping: remove a block, greedily re-cover newly uncovered t-subsets,
       accept if net reduction."""
    blocks = [list(b) for b in blocks]
    coverage = _build_coverage(blocks, v, t)

    improved = True
    while improved and time.time() < deadline:
        improved = False

        # Phase A: Remove redundant blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            newly_uncovered = []
            for ts in itertools.combinations(blocks[bi], t):
                ts = tuple(sorted(ts))
                cov = coverage.get(ts, set())
                if len(cov) == 1 and bi in cov:
                    newly_uncovered.append(ts)
            if not newly_uncovered:
                _remove_block(blocks, bi, coverage, t)
                improved = True
                break

        if improved or time.time() >= deadline:
            continue

        # Phase B: Try swap - remove block, re-cover with new blocks
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Save state
            saved_blocks = [list(b) for b in blocks]
            saved_coverage = {ts: set(s) for ts, s in coverage.items()}

            # Remove block bi
            newly_uncovered = _remove_block(blocks, bi, coverage, t)
            if not newly_uncovered:
                # Shouldn't happen (handled in Phase A), but be safe
                blocks = saved_blocks
                coverage = saved_coverage
                continue

            # Try to re-cover newly_uncovered with new blocks
            uncovered = set(newly_uncovered)
            new_blocks_added = 0
            max_new_blocks = 2  # Only accept if we add at most 1 new block (net -1) or 2 (net 0, skip)

            while uncovered and new_blocks_added < max_new_blocks:
                if time.time() >= deadline:
                    break
                seed = next(iter(uncovered))
                block = list(seed)
                block = _greedy_fill(block, uncovered, v, k, t)
                block.sort()
                _add_block(blocks, block, coverage, t)
                new_blocks_added += 1
                # Update uncovered
                for ts in itertools.combinations(block, t):
                    uncovered.discard(tuple(sorted(ts)))

            # Check if we actually reduced block count
            if len(blocks) < len(saved_blocks) and not uncovered:
                improved = True
                break
            else:
                # Revert
                blocks = saved_blocks
                coverage = saved_coverage

    return blocks


def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin before 120s limit
    rng = random.Random(42)

    # Phase 1: Greedy cover with randomization, multiple restarts
    best_blocks = None
    best_count = float('inf')

    # First greedy run
    blocks = _greedy_cover(v, k, t, rng, deadline)
    best_blocks = blocks
    best_count = len(blocks)

    # Try more greedy restarts if time permits
    restarts = 0
    while time.time() < deadline - 40 and restarts < 8:
        blocks = _greedy_cover(v, k, t, rng, deadline)
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
        restarts += 1

    # Phase 2: Local search to reduce block count
    if time.time() < deadline - 10:
        best_blocks = _local_search(best_blocks, v, k, t, deadline - 3)

    return best_blocks
# EVOLVE-BLOCK-END
