"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    total_tsubsets = len(all_tsubsets)

    # Track uncovered t-subsets
    uncovered = set(all_tsubsets)
    blocks = []

    while uncovered:
        if time.time() > deadline:
            break

        # Pick the first uncovered t-subset to seed a new block
        seed = min(uncovered)
        block = list(seed)
        block_set = set(block)

        # Build the rest of the block greedily
        while len(block) < k:
            if time.time() > deadline:
                break

            # For each candidate x not in block, compute gain
            # gain(x) = number of uncovered t-subsets that contain x and (t-1) elements from block
            t_minus_1_subsets = list(itertools.combinations(block, t - 1)) if t >= 2 else [()]

            best_x = None
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                gain = 0
                for rest in t_minus_1_subsets:
                    rest_set = set(rest)
                    if x in rest_set:
                        continue
                    candidate = tuple(sorted(rest + (x,)))
                    if candidate in uncovered:
                        gain += 1

                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None:
                remaining = [x for x in range(v) if x not in block_set]
                if remaining:
                    best_x = random.choice(remaining)
                else:
                    break

            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            break

        block.sort()
        blocks.append(tuple(block))

        # Remove all t-subsets contained in this block from uncovered
        for ts in itertools.combinations(block, t):
            uncovered.discard(ts)

    # Local search: try to remove redundant blocks using coverage counts
    if blocks and time.time() < deadline - 3:
        # Build coverage count for each t-subset
        cover_count = defaultdict(int)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cover_count[ts] += 1

        n = len(blocks)
        changed = True
        while changed and time.time() < deadline - 1:
            changed = False
            for i in range(n - 1, -1, -1):
                if time.time() > deadline - 1:
                    break
                block_i = blocks[i]
                redundant = True
                for ts in itertools.combinations(block_i, t):
                    if cover_count[ts] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts in itertools.combinations(block_i, t):
                        cover_count[ts] -= 1
                    blocks.pop(i)
                    n -= 1
                    changed = True
                    break  # Restart scan since indices shifted

    return [list(b) for b in blocks]
# EVOLVE-BLOCK-END
