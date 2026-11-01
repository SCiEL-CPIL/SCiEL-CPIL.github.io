"""Improved greedy covering design with redundancy removal and time guard."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using an improved greedy algorithm
    with redundancy removal and a deadline guard.
    """
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = random.Random(42)

    # All t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_t_subsets)

    # Map each t-subset to an index for fast lookup
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # For each point, precompute which t-subsets it participates in
    # point_to_subsets[p] = list of indices of t-subsets containing p
    point_to_subsets = [[] for _ in range(v)]
    for idx, sub in enumerate(all_t_subsets):
        for p in sub:
            point_to_subsets[p].append(idx)

    # uncovered: set of indices of uncovered t-subsets
    uncovered = set(range(n_subsets))

    blocks = []

    while uncovered and time.time() < deadline:
        # Pick the most constrained uncovered t-subset as seed
        # (the one with the fewest candidate blocks that could cover it)
        # For simplicity, pick the first uncovered subset (deterministic)
        # but we can also pick randomly for diversity
        seed_subset = min(uncovered)
        block = list(all_t_subsets[seed_subset])

        # Greedily add points that maximize coverage of uncovered t-subsets
        while len(block) < k and time.time() < deadline:
            block_set = set(block)
            best_point = None
            best_gain = -1

            # For each candidate point not in block, compute gain
            # Gain = number of uncovered t-subsets that would be covered
            # A t-subset is covered if all its t elements are in the block
            # When we add point x, the new t-subsets covered are those
            # that contain x and have (t-1) other elements already in block

            # Precompute: for each point x, which t-subsets contain x?
            # We need t-subsets where x is present and the other t-1 elements are in block

            for x in range(v):
                if x in block_set:
                    continue

                # Count uncovered t-subsets that contain x and have
                # at least (t-1) other elements in block
                gain = 0
                for sub_idx in point_to_subsets[x]:
                    if sub_idx not in uncovered:
                        continue
                    sub = all_t_subsets[sub_idx]
                    # Check if all other elements of sub are in block
                    count_in_block = 0
                    for elem in sub:
                        if elem == x:
                            continue
                        if elem in block_set:
                            count_in_block += 1
                    if count_in_block == t - 1:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_point = x

            if best_point is None or best_gain <= 0:
                # No point helps; break
                break

            block.append(best_point)

        if len(block) < k:
            # Fill remaining with arbitrary points
            block_set = set(block)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)

        block.sort()
        blocks.append(block)

        # Remove covered t-subsets from uncovered
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            uncovered.discard(idx)

    # If we ran out of time but haven't covered everything, we still return
    # whatever blocks we have. But we need to ensure validity.
    # Let's check if all t-subsets are covered. If not, we need to add more blocks.
    # However, given time constraints, we'll do a final greedy pass if needed.

    # Check coverage
    if uncovered:
        # Do a quick greedy fill for remaining uncovered subsets
        while uncovered and time.time() < deadline - 5:
            seed_subset = min(uncovered)
            block = list(all_t_subsets[seed_subset])

            while len(block) < k:
                block_set = set(block)
                best_point = None
                best_gain = -1

                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub_idx in point_to_subsets[x]:
                        if sub_idx not in uncovered:
                            continue
                        sub = all_t_subsets[sub_idx]
                        count_in_block = 0
                        for elem in sub:
                            if elem == x:
                                continue
                            if elem in block_set:
                                count_in_block += 1
                        if count_in_block == t - 1:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x

                if best_point is None:
                    break
                block.append(best_point)

            if len(block) < k:
                block_set = set(block)
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)

            block.sort()
            blocks.append(block)

            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                uncovered.discard(idx)

    # Redundancy removal: try to remove blocks that don't uniquely cover any t-subset
    # A block is redundant if for every t-subset it covers, some other block also covers it
    if time.time() < deadline - 10:
        # Build coverage map: for each t-subset, which blocks cover it
        subset_blocks = defaultdict(set)
        for bi, block in enumerate(blocks):
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                subset_blocks[idx].add(bi)

        # Try to remove blocks
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for bi in range(len(blocks)):
                # Check if block bi is redundant
                redundant = True
                for sub in itertools.combinations(blocks[bi], t):
                    idx = subset_to_idx[sub]
                    if len(subset_blocks[idx]) <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block bi
                    for sub in itertools.combinations(blocks[bi], t):
                        idx = subset_to_idx[sub]
                        subset_blocks[idx].discard(bi)
                    blocks.pop(bi)
                    changed = True
                    break  # restart scan since indices changed

    return blocks
# EVOLVE-BLOCK-END
