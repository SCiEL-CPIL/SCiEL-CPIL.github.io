"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Greedy covering with deadline guard.
    # Strategy: pick the uncovered t-subset with maximum potential coverage,
    # then build a block around it by adding points that cover the most
    # remaining uncovered t-subsets.
    deadline = time.time() + 115.0  # safety margin before 120s limit

    all_points = list(range(v))
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []

    # Precompute for speed: use a helper to count gain efficiently
    while uncovered:
        if time.time() > deadline:
            break

        # Pick the uncovered t-subset that has the most "potential"
        # (i.e., appears in the most candidate blocks we could form)
        # For efficiency, just pick the first uncovered one (lexicographic min)
        # but try a small random sample to find a better seed
        seed = list(min(uncovered))

        # Build block greedily
        block_set = set(seed)
        block = list(seed)

        while len(block) < k:
            if time.time() > deadline:
                break

            best_x = -1
            best_gain = -1
            # Try each candidate point
            for x in all_points:
                if x in block_set:
                    continue
                # Count how many uncovered t-subsets would be covered
                # by adding x to the current block
                gain = 0
                # A t-subset containing x and (t-1) elements from block
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(rest + (x,)))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x == -1:
                break
            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            break

        block.sort()
        blocks.append(block)
        # Remove all t-subsets of this block from uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks
# EVOLVE-BLOCK-END
