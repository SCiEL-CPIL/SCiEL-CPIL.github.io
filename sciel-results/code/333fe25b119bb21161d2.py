"""Stronger covering design C(19,7,4) solver.

Strategy:
1. Fast greedy construction with incremental coverage counting.
2. Redundant block removal.
3. Local search: try to remove blocks, and replace blocks to enable further removals.
4. Deadline guard to return best result before timeout.
"""

import itertools
import time
import random
from collections import defaultdict, Counter


def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1}."""
    return set(itertools.combinations(range(v), t))


def _block_subs(block, t):
    """Return the set of t-subsets covered by a block."""
    return set(itertools.combinations(block, t))


def _greedy_construct(v, k, t, deadline):
    """Greedy covering algorithm with fast gain computation."""
    all_subs = _all_t_subsets(v, t)
    uncovered = set(all_subs)
    blocks = []

    while uncovered and time.time() < deadline:
        # Start with the lexicographically smallest uncovered t-subset
        min_sub = min(uncovered)
        block = list(min_sub)
        block_set = set(block)

        # Greedily add points
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue
                # Compute gain: number of uncovered t-subsets containing x and (t-1) from block
                gain = 0
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

        block.sort()
        blocks.append(block)

        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)

    return blocks, uncovered


def _remove_redundant_blocks(blocks, v, k, t, deadline):
    """Remove blocks that don't cover any unique t-subset."""
    if not blocks:
        return blocks

    # Count how many blocks cover each t-subset
    sub_count = defaultdict(int)
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    # A block is redundant if every t-subset it covers is also covered by another block
    kept = []
    for block in blocks:
        redundant = True
        for sub in itertools.combinations(block, t):
            if sub_count[sub] == 1:
                redundant = False
                break
        if not redundant:
            kept.append(block)

    return kept


def _local_search(blocks, v, k, t, deadline):
    """Try to improve the covering by removing or replacing blocks."""
    all_subs = _all_t_subsets(v, t)
    total_subs = len(all_subs)

    # Count coverage
    sub_count = Counter()
    for block in blocks:
        for sub in itertools.combinations(block, t):
            sub_count[sub] += 1

    # First, remove any redundant blocks
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(blocks)):
            if time.time() > deadline:
                break
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                changed = True
                break

    # Now try to replace blocks to enable further removals
    random.seed(42)
    improved = True
    max_replacements = 2000
    replacements = 0

    while improved and time.time() < deadline and replacements < max_replacements:
        improved = False
        replacements += 1

        # Try removing each block and see if coverage is still complete
        for i in range(len(blocks)):
            if time.time() > deadline:
                return blocks
            block = blocks[i]
            needed = False
            for sub in itertools.combinations(block, t):
                if sub_count[sub] == 1:
                    needed = True
                    break
            if not needed:
                for sub in itertools.combinations(block, t):
                    sub_count[sub] -= 1
                blocks.pop(i)
                improved = True
                break

        if improved:
            continue

        # Try replacing a block with a better one
        if len(blocks) > 0:
            # Pick a random block to try replacing
            idx = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx]
            old_subs = _block_subs(old_block, t)

            # Count how many of old_block's subsets are uniquely covered by it
            unique_count = 0
            for sub in old_subs:
                if sub_count[sub] == 1:
                    unique_count += 1

            # Generate several random candidate blocks
            best_candidate = None
            best_score = -1

            for _ in range(20):
                new_block = sorted(random.sample(range(v), k))
                new_subs = _block_subs(new_block, t)

                # Compute score: how many new subsets does this cover that aren't already covered?
                # And how many unique subsets would we lose?
                gained = 0
                for sub in new_subs:
                    if sub_count[sub] == 0:
                        gained += 1

                lost = 0
                for sub in old_subs:
                    if sub_count[sub] == 1 and sub not in new_subs:
                        lost += 1

                score = gained - lost * 10  # Heavily penalize losing unique coverage

                if score > best_score:
                    best_score = score
                    best_candidate = new_block

            if best_candidate is not None and best_score > 0:
                # Apply the replacement
                for sub in old_subs:
                    sub_count[sub] -= 1
                for sub in _block_subs(best_candidate, t):
                    sub_count[sub] += 1
                blocks[idx] = best_candidate
                improved = True

    return blocks


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety

    # Phase 1: Greedy construction
    blocks, uncovered = _greedy_construct(v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 2: Remove redundant blocks
    blocks = _remove_redundant_blocks(blocks, v, k, t, deadline)

    if time.time() > deadline:
        return blocks

    # Phase 3: Local search - try to find better blocks
    blocks = _local_search(blocks, v, k, t, deadline)

    return blocks
