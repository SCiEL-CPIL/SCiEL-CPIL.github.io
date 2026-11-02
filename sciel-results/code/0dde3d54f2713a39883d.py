# EVOLVE-BLOCK-START
import itertools
import time
import heapq
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)

    # For each point, precompute which t-subsets contain it
    point_tsubs = [set() for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for pt in ts:
            point_tsubs[pt].add(i)

    # Track coverage count for each t-subset
    covered = bytearray(n_tsubs)  # 0 or 1
    cover_count = [0] * n_tsubs

    blocks = []
    block_set = set()

    # Helper: check if a block is valid (k distinct points in 0..v-1)
    def is_valid_block(block):
        if len(block) != k:
            return False
        if len(set(block)) != k:
            return False
        if any(x < 0 or x >= v for x in block):
            return False
        return True

    # Helper: get the set of t-subset indices covered by a block
    def block_tsub_indices(block):
        """Return set of t-subset indices covered by this block."""
        result = set()
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                result.add(idx)
        return result

    # Helper: compute gain of adding point x to current block
    def compute_gain(block, x):
        """How many currently-uncovered t-subsets would be covered by adding x to block."""
        gain = 0
        for rest in itertools.combinations(block, t - 1):
            ts = tuple(sorted(rest + (x,)))
            idx = tsub_to_idx.get(ts)
            if idx is not None and not covered[idx]:
                gain += 1
        return gain

    # Main greedy loop
    while True:
        if time.time() > deadline:
            break

        # Find uncovered t-subsets
        uncovered_indices = [i for i in range(n_tsubs) if not covered[i]]
        if not uncovered_indices:
            break

        # Pick the most constrained uncovered t-subset (one with fewest possible blocks)
        # For simplicity, pick the first uncovered one (lexicographically smallest)
        target_idx = uncovered_indices[0]
        target_tsub = all_tsubs[target_idx]

        # Build a block starting from the target t-subset
        block = list(target_tsub)
        block_set_frozen = frozenset(block)

        while len(block) < k:
            if time.time() > deadline:
                break

            best_point = None
            best_gain = -1

            candidates = [x for x in range(v) if x not in block]
            for x in candidates:
                g = compute_gain(block, x)
                if g > best_gain:
                    best_gain = g
                    best_point = x
                elif g == best_gain and best_point is not None and x < best_point:
                    best_point = x

            if best_point is None:
                break
            block.append(best_point)

        block.sort()
        block_tuple = tuple(block)

        if not is_valid_block(block):
            continue

        if block_tuple in block_set:
            continue

        # Check if this block adds any new coverage
        new_cover = False
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None and not covered[idx]:
                new_cover = True
                break

        if not new_cover:
            continue

        # Add the block
        block_set.add(block_tuple)
        blocks.append(list(block))

        # Update coverage
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                covered[idx] = 1
                cover_count[idx] += 1

    # Post-processing: remove redundant blocks
    if time.time() < deadline - 10:
        # Try to remove each block if all its t-subsets are still covered by others
        i = 0
        while i < len(blocks):
            if time.time() > deadline:
                break

            block = blocks[i]
            block_tsubs = list(itertools.combinations(block, t))

            # Check if removing this block leaves all its t-subsets covered
            removable = True
            for ts in block_tsubs:
                idx = tsub_to_idx.get(ts)
                if idx is not None and cover_count[idx] <= 1:
                    removable = False
                    break

            if removable:
                # Remove this block and update counts
                for ts in block_tsubs:
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            covered[idx] = 0
                blocks.pop(i)
                # Don't increment i, check the same position again
            else:
                i += 1

    # Verify completeness (should always be true if we built it correctly)
    # But let's make sure we have a valid covering
    final_covered = bytearray(n_tsubs)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx.get(ts)
            if idx is not None:
                final_covered[idx] = 1

    # If something went wrong, fall back to a simple complete covering
    if any(not final_covered[i] for i in range(n_tsubs)):
        # Emergency: add blocks to cover remaining t-subsets
        for i in range(n_tsubs):
            if not final_covered[i]:
                ts = all_tsubs[i]
                # Extend this t-subset to a k-subset
                block = list(ts)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block:
                        block.append(x)
                block.sort()
                blocks.append(block)
                for t2 in itertools.combinations(block, t):
                    idx2 = tsub_to_idx.get(t2)
                    if idx2 is not None:
                        final_covered[idx2] = 1

    return blocks
# EVOLVE-BLOCK-END
