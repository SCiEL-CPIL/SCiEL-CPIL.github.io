"""Improved greedy covering with simulated-annealing polish and time guard."""
import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}

    # Greedy construction
    uncovered = set(range(n_tsubs))
    blocks = []

    while uncovered:
        if time.time() > deadline - 10.0:
            break

        # Pick the uncovered t-subset with minimum index (deterministic)
        first_uncovered = min(uncovered)
        block = list(all_tsubs[first_uncovered])

        while len(block) < k:
            block_set = set(block)
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered by adding x
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    new_tsub = tuple(sorted(rest + (x,)))
                    idx = tsub_index.get(new_tsub, -1)
                    if idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            block.append(best_x)

        block.sort()
        blocks.append(block)

        # Remove covered t-subsets
        for ts in itertools.combinations(block, t):
            idx = tsub_index[ts]
            if idx in uncovered:
                uncovered.discard(idx)

    # Simulated annealing polish
    if uncovered and time.time() < deadline - 5.0:
        blocks_set = set(tuple(b) for b in blocks)
        current_blocks = list(blocks)
        current_uncovered = set(uncovered)
        best_blocks = list(current_blocks)
        best_uncovered = set(current_uncovered)
        best_score = len(best_blocks)

        T = 10.0
        T_min = 0.1
        cooling = 0.9995
        rng = random.Random(42)

        while time.time() < deadline - 3.0 and T > T_min:
            # Try to modify a random block
            if current_blocks:
                idx = rng.randrange(len(current_blocks))
                old_block = list(current_blocks[idx])
                new_block = list(old_block)

                # Randomly swap a point in the block
                if new_block:
                    pos = rng.randrange(len(new_block))
                    old_point = new_block[pos]
                    candidates = [x for x in range(v) if x not in new_block]
                    if candidates:
                        new_point = rng.choice(candidates)
                        new_block[pos] = new_point
                        new_block.sort()

                    # Compute change in uncovered set
                    old_covered = set(tsub_index[ts] for ts in itertools.combinations(old_block, t))
                    new_covered = set(tsub_index[ts] for ts in itertools.combinations(new_block, t))

                    # Uncovered that were covered by old but not new
                    lost = old_covered - new_covered
                    # Uncovered that are now covered by new but weren't by old
                    gained = new_covered - old_covered

                    # Check if new_block is already in blocks
                    if tuple(new_block) in blocks_set:
                        T *= cooling
                        continue

                    delta_uncovered = len(lost) - len(gained)
                    if delta_uncovered <= 0 or rng.random() < (10 ** (-delta_uncovered / T)):
                        # Accept
                        blocks_set.discard(tuple(old_block))
                        blocks_set.add(tuple(new_block))
                        current_blocks[idx] = new_block
                        current_uncovered = (current_uncovered | lost) - gained

                        if len(current_uncovered) < len(best_uncovered):
                            best_uncovered = set(current_uncovered)
                            best_blocks = list(current_blocks)
                            best_score = len(best_blocks)
                    else:
                        # Reject
                        pass

            T *= cooling

        if len(best_uncovered) < len(uncovered):
            blocks = best_blocks
            uncovered = best_uncovered

    return blocks
