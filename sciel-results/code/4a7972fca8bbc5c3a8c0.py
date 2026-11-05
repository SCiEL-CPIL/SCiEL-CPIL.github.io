"""Improved covering design C(24,7,3) via faster greedy with numpy acceleration."""
import time
import itertools
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s kill

    # Precompute all t-subsets as integer indices
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

        # Start with the lexicographically smallest uncovered t-subset
        first_idx = np.argmax(uncovered)  # first True
        block = list(all_tsubs[first_idx])

        # Build the block greedily
        while len(block) < k:
            if time.time() > deadline:
                break

            block_set = set(block)
            # Candidates: points not in block
            candidates = [x for x in range(v) if x not in block_set]

            # For each candidate, compute how many uncovered t-subsets it would cover
            # A t-subset is covered by adding x if it contains x and (t-1) points from block
            # and is currently uncovered
            best_x = None
            best_gain = -1

            # Precompute (t-1)-subsets of block for efficiency
            block_tminus1 = list(itertools.combinations(block, t - 1))

            for x in candidates:
                gain = 0
                for rest in block_tminus1:
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # Pick any remaining point
                best_x = candidates[0] if candidates else v - 1

            block.append(best_x)

        if len(block) < k:
            break  # timeout during block construction

        block.sort()
        blocks.append(block)

        # Update uncovered: mark all t-subsets of this block as covered
        for ts in itertools.combinations(block, t):
            idx = tsub_to_idx[ts]
            uncovered[idx] = False

    return blocks
