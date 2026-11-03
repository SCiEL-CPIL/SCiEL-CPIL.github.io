"""Improved greedy covering with better block construction and local search."""
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)

    # Map each t-subset to an index for fast lookup
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # Coverage count: how many blocks cover each t-subset
    coverage = [0] * num_t_subsets
    uncovered_count = num_t_subsets

    blocks = []

    # Precompute for each point x, the list of t-subsets that contain x
    # For efficiency, we'll compute on the fly but cache per-block

    # Precompute combinations of t-1 for gain calculation
    # We need to know, for a partial block B and candidate x,
    # how many uncovered t-subsets are formed by {x} + (t-1)-subset of B

    # For speed: precompute all (t-1)-subsets of each possible block prefix? No, too many.
    # Instead, for each partial block, compute the set of (t-1)-subsets it contains.

    def time_remaining():
        return deadline - time.time()

    def greedy_block(uncovered_set, partial=None):
        """Build one block greedily to maximize coverage of uncovered t-subsets."""
        if partial is None:
            # Start with the t-subset that has the least coverage (hardest to cover)
            # Or just pick the first uncovered one
            min_cov = min(coverage)
            # Find an uncovered t-subset with minimum coverage
            best_start = None
            best_cov = float('inf')
            for s in uncovered_set:
                idx = t_subset_to_idx[s]
                if coverage[idx] < best_cov:
                    best_cov = coverage[idx]
                    best_start = s
                    if best_cov == 0:
                        break
            if best_start is None:
                # All have same coverage, just pick first
                best_start = next(iter(uncovered_set))
            block = list(best_start)
        else:
            block = list(partial)

        block_set = set(block)

        while len(block) < k and time_remaining() > 5:
            # For each candidate x not in block, compute gain
            # gain(x) = number of uncovered t-subsets that contain x and (t-1) elements from block
            best_x = None
            best_gain = -1

            # Precompute (t-1)-subsets of current block
            t_minus_1_subsets = list(itertools.combinations(block, t - 1))

            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for sub in t_minus_1_subsets:
                    # Check if {x} + sub forms an uncovered t-subset
                    new_subset = tuple(sorted(sub + (x,)))
                    if new_subset in uncovered_set:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            if best_x is None or best_gain <= 0:
                # No more useful points to add; break
                break

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        return block

    # Main greedy loop
    while uncovered_count > 0 and time_remaining() > 10:
        # Get set of uncovered t-subsets
        uncovered_set = set()
        for i in range(num_t_subsets):
            if coverage[i] == 0:
                uncovered_set.add(all_t_subsets[i])

        if not uncovered_set:
            break

        # Build a block
        block = greedy_block(uncovered_set)
        blocks.append(block)

        # Update coverage
        block_t_subsets = list(itertools.combinations(block, t))
        for s in block_t_subsets:
            idx = t_subset_to_idx[s]
            if coverage[idx] == 0:
                uncovered_count -= 1
            coverage[idx] += 1

    # Local search phase: try to improve by replacing blocks
    if time_remaining() > 20:
        # Try random restarts with different initial orderings
        best_blocks = list(blocks)
        best_count = len(blocks)

        num_restarts = min(5, max(1, int(time_remaining() // 30)))
        for restart in range(num_restarts):
            if time_remaining() < 25:
                break

            # Shuffle the order of uncovered t-subsets processing
            random.seed(42 + restart)
            coverage2 = [0] * num_t_subsets
            uncovered_count2 = num_t_subsets
            blocks2 = []

            # Precompute a shuffled order of t-subsets
            shuffled_indices = list(range(num_t_subsets))
            random.shuffle(shuffled_indices)

            while uncovered_count2 > 0 and time_remaining() > 20:
                # Find the first uncovered t-subset in shuffled order
                start_subset = None
                for idx in shuffled_indices:
                    if coverage2[idx] == 0:
                        start_subset = all_t_subsets[idx]
                        break
                if start_subset is None:
                    break

                block = list(start_subset)
                block_set = set(block)

                while len(block) < k and time_remaining() > 15:
                    t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                    best_x = None
                    best_gain = -1

                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for sub in t_minus_1_subsets:
                            new_subset = tuple(sorted(sub + (x,)))
                            if coverage2[t_subset_to_idx[new_subset]] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x

                    if best_x is None or best_gain <= 0:
                        break

                    block.append(best_x)
                    block_set.add(best_x)

                block.sort()
                blocks2.append(block)

                for s in itertools.combinations(block, t):
                    idx = t_subset_to_idx[s]
                    if coverage2[idx] == 0:
                        uncovered_count2 -= 1
                    coverage2[idx] += 1

            if len(blocks2) < best_count:
                best_blocks = blocks2
                best_count = len(blocks2)

        blocks = best_blocks

    return blocks
