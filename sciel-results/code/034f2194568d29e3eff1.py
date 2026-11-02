"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import math


def _precompute(v, k, t):
    """Precompute all t-subsets and index mapping for fast coverage checks."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    tset_index = {ts: i for i, ts in enumerate(all_tsubsets)}
    n_t = len(all_tsubsets)
    return all_tsubsets, tset_index, n_t


def _block_tsets(block, tset_index):
    """Return list of indices of t-subsets contained in block."""
    return [tset_index[ts] for ts in itertools.combinations(block, t)]


def _greedy_construct(v, k, t, all_tsubsets, tset_index, n_t, deadline, rng=None):
    """
    Greedy set cover: repeatedly pick the block that covers the most uncovered t-subsets.
    Uses a randomized tie-breaking and a local improvement pass.
    """
    if rng is None:
        rng = random.Random()

    # uncovered is a boolean array of length n_t
    uncovered = bytearray([1]) * n_t
    blocks = []
    n_blocks = 0

    # Precompute for each point, the list of t-subsets containing it (for fast gain calc)
    # point_tsets[p] = list of indices of t-subsets that contain point p
    point_tsets = [[] for _ in range(v)]
    for idx, ts in enumerate(all_tsubsets):
        for p in ts:
            point_tsets[p].append(idx)

    # For a partial block (set of points), the t-subsets it can cover are those
    # where all t points are in the block. We track which t-subsets are "almost covered"
    # i.e., have t-1 points in the block, and adding one more would complete them.

    while any(uncovered):
        if time.time() > deadline:
            break

        # Greedy: build a block by starting with the most uncovered t-subset's points,
        # then iteratively adding the point that maximizes new coverage.
        # Find an uncovered t-subset to seed the block
        # Pick a random uncovered t-subset for diversity
        seed_idx = rng.randrange(n_t)
        while not uncovered[seed_idx]:
            seed_idx = rng.randrange(n_t)
        seed_tset = all_tsubsets[seed_idx]
        block = list(seed_tset)

        # Track which t-subsets are covered by the current block
        block_covered = bytearray([0]) * n_t  # 1 if this t-subset is covered by current block
        # Initialize with t-subsets fully contained in seed
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            block_covered[idx] = 1

        # Iteratively add points
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Compute gain: how many currently-uncovered t-subsets would become covered
                # A t-subset is newly covered if it contains x and the other t-1 points are in block
                # and it's currently uncovered
                gain = 0
                # For each (t-1)-subset of block, check if (that subset + x) is a t-subset
                # that is uncovered
                for rest in itertools.combinations(block, t - 1):
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in tset_index:
                        idx = tset_index[candidate]
                        if uncovered[idx] and not block_covered[idx]:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
                elif gain == best_gain and best_point != -1:
                    # Random tie-break
                    if rng.random() < 0.5:
                        best_point = x

            if best_point == -1:
                break
            block.append(best_point)
            block.sort()

            # Update block_covered for new t-subsets containing best_point
            for ts in itertools.combinations(block, t):
                if best_point in ts:
                    idx = tset_index[ts]
                    if not block_covered[idx]:
                        block_covered[idx] = 1

        block.sort()
        # Only add block if it covers at least one new t-subset
        new_covers = 0
        for idx in range(n_t):
            if block_covered[idx] and uncovered[idx]:
                new_covers += 1
        if new_covers == 0:
            # This block doesn't help; try a different seed
            continue

        blocks.append(block)
        n_blocks += 1

        # Update uncovered
        for idx in range(n_t):
            if block_covered[idx]:
                uncovered[idx] = 0

    return blocks


def _local_search(blocks, v, k, t, all_tsubsets, tset_index, n_t, deadline, rng):
    """
    1-for-1 swap local search: try replacing one block with a different block
    if it doesn't break coverage and (optionally) allows removing another block.
    Here we do: try to remove a block if all its t-subsets are covered by others.
    Then try to swap: replace block i with a new block that covers more.
    """
    if not blocks:
        return blocks

    # Build coverage count: for each t-subset, how many blocks cover it
    cover_count = [0] * n_t
    block_tsets = []
    for block in blocks:
        tsets = _block_tsets(block, tset_index)
        block_tsets.append(tsets)
        for idx in tsets:
            cover_count[idx] += 1

    # Phase 1: Remove redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            # Check if block i is redundant: all its t-subsets are covered by >= 2 blocks
            redundant = True
            for idx in block_tsets[i]:
                if cover_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove block i
                for idx in block_tsets[i]:
                    cover_count[idx] -= 1
                blocks.pop(i)
                block_tsets.pop(i)
                changed = True
                break

    # Phase 2: Try to swap blocks - replace a block with one that covers more unique t-subsets
    # This is expensive; do a limited number of iterations
    max_swaps = 50
    swaps_done = 0
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not blocks:
            break
        i = rng.randrange(len(blocks))
        # Try to find a better replacement for block i
        # The t-subsets that ONLY block i covers must still be covered
        unique_to_i = []
        for idx in block_tsets[i]:
            if cover_count[idx] == 1:
                unique_to_i.append(idx)
        if not unique_to_i:
            # Block i is redundant, remove it
            for idx in block_tsets[i]:
                cover_count[idx] -= 1
            blocks.pop(i)
            block_tsets.pop(i)
            swaps_done += 1
            continue

        # Find a block that covers all unique_to_i and adds some new coverage
        best_j = -1
        best_extra = -1
        for j in range(len(blocks)):
            if j == i:
                continue
            # Check if block j covers all unique_to_i
            j_set = set(block_tsets[j])
            if all(idx in j_set for idx in unique_to_i):
                # Count how many t-subsets in block j are not in block i
                i_set = set(block_tsets[i])
                extra = len(j_set - i_set)
                if extra > best_extra:
                    best_extra = extra
                    best_j = j
        if best_j != -1 and best_extra > 0:
            # Replace block i with block best_j
            # Remove coverage of block i
            for idx in block_tsets[i]:
                cover_count[idx] -= 1
            # Add coverage of block best_j
            for idx in block_tsets[best_j]:
                cover_count[idx] += 1
            blocks[i] = blocks[best_j]
            block_tsets[i] = block_tsets[best_j]
            # Remove duplicate block best_j
            blocks.pop(best_j)
            block_tsets.pop(best_j)
            swaps_done += 1

    return blocks


def _verify_covering(blocks, v, k, t, all_tsubsets, tset_index):
    """Check if every t-subset is covered by at least one block."""
    covered = bytearray([0]) * len(all_tsubsets)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tset_index[ts]
            covered[idx] = 1
    return all(covered)


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin before 120s limit

    # Precompute
    all_tsubsets, tset_index, n_t = _precompute(v, k, t)

    # If the problem is small enough, we can do multiple restarts
    # Estimate work per greedy run
    rng = random.Random(42)

    best_blocks = None
    best_count = float('inf')

    # Multi-start: run greedy multiple times with different seeds
    # Number of restarts depends on time budget
    # First run to get a baseline
    blocks1 = _greedy_construct(v, k, t, all_tsubsets, tset_index, n_t, deadline, rng)
    if _verify_covering(blocks1, v, k, t, all_tsubsets, tset_index):
        best_blocks = blocks1
        best_count = len(blocks1)

    # Local search on the first solution
    if best_blocks is not None:
        improved = _local_search(list(best_blocks), v, k, t, all_tsubsets, tset_index, n_t, deadline, rng)
        if _verify_covering(improved, v, k, t, all_tsubsets, tset_index):
            if len(improved) < best_count:
                best_blocks = improved
                best_count = len(improved)

    # More restarts if time permits
    max_restarts = 20
    for restart in range(1, max_restarts):
        if time.time() > deadline - 15:  # Leave 15s for final local search
            break
        rng2 = random.Random(42 + restart * 7919)
        blocks_r = _greedy_construct(v, k, t, all_tsubsets, tset_index, n_t, deadline, rng2)
        if _verify_covering(blocks_r, v, k, t, all_tsubsets, tset_index):
            if len(blocks_r) < best_count:
                best_blocks = blocks_r
                best_count = len(blocks_r)

    # Final local search on the best solution
    if best_blocks is not None and time.time() < deadline - 5:
        final = _local_search(list(best_blocks), v, k, t, all_tsubsets, tset_index, n_t, deadline, random.Random(12345))
        if _verify_covering(final, v, k, t, all_tsubsets, tset_index):
            if len(final) < best_count:
                best_blocks = final
                best_count = len(final)

    if best_blocks is None:
        # Fallback: should not happen, but return empty if something went wrong
        return []

    return best_blocks
# EVOLVE-BLOCK-END
