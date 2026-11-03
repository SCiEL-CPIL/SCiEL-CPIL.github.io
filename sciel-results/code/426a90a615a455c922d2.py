"""Greedy covering with vectorized gain computation and deadline guard."""
import itertools
import time
import numpy as np


def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)

    # Map each t-subset to an index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # uncovered: boolean array, True if not yet covered
    uncovered = np.ones(n_tsubs, dtype=bool)

    blocks = []

    while True:
        if time.time() > deadline:
            break
        if not uncovered.any():
            break

        # Pick the first uncovered t-subset to start a block
        first_idx = int(np.argmax(uncovered))
        first_tsub = all_tsubs[first_idx]
        block = list(first_tsub)
        block_set = set(block)

        # Greedily add points to maximize coverage of uncovered t-subsets
        remaining_slots = k - t
        for _ in range(remaining_slots):
            if time.time() > deadline:
                break

            best_gain = -1
            best_point = -1

            # For each candidate point not in block, compute how many new
            # uncovered t-subsets it would cover when added to the block
            for x in range(v):
                if x in block_set:
                    continue

                # A t-subset is covered by adding x to block iff it contains x
                # and the other (t-1) elements are in block.
                # So we need to count uncovered t-subsets that contain x and
                # (t-1) elements from block.
                #
                # Efficient approach: for each (t-1)-subset of block, the t-subset
                # formed by adding x must be in uncovered.
                count = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts, -1)
                    if idx >= 0 and uncovered[idx]:
                        count += 1

                if count > best_gain:
                    best_gain = count
                    best_point = x

            if best_point == -1:
                break
            block.append(best_point)
            block_set.add(best_point)

        block.sort()
        blocks.append(block)

        # Mark all t-subsets within this block as covered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = False

        # Check if we've covered everything
        if not uncovered.any():
            break

    return blocks
