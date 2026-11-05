"""Covering design C(v,k,t) via greedy + iterative local search on uncovered count."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def _greedy_cover(v, k, t, rng):
    """Greedy covering: repeatedly add the block with max marginal coverage."""
    # Represent uncovered t-subsets as a set of frozensets for O(1) membership.
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    if not uncovered:
        return blocks
    while uncovered:
        # Candidate block: pick a random uncovered t-subset as seed, then greedily extend.
        seed = tuple(rng.choice(list(uncovered)))  # one uncovered t-subset
        block_set = set(seed)
        block = list(seed)
        while len(block) < k:
            best_x = -1
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                # Count uncovered t-subsets that contain x and t-1 points from block
                for rest in itertools.combinations(block, t - 1):
                    if tuple(sorted(rest + (x,))) in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x < 0:
                # pick any unused point
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
            block.append(best_x)
            block_set.add(best_x)
        block.sort()
        blocks.append(block)
        # Remove all t-subsets of this block from uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
    return blocks


def _count_uncovered(blocks, v, t):
    """Return the set of uncovered t-subsets given a list of blocks."""
    covered = set()
    for b in blocks:
        for sub in itertools.combinations(b, t):
            covered.add(sub)
    all_subsets = set(itertools.combinations(range(v), t))
    return all_subsets - covered


def _best_replacement_block(block, uncovered, v, k, t, rng):
    """Find the k-subset that maximizes coverage of uncovered t-subsets,
    with a random tie-break to encourage diversity."""
    # Start from a random uncovered t-subset as a seed for the candidate block.
    if not uncovered:
        return None
    seed = tuple(rng.choice(list(uncovered)))
    cand = list(seed)
    cand_set = set(seed)
    while len(cand) < k:
        best_x = -1
        best_gain = -1
        for x in range(v):
            if x in cand_set:
                continue
            gain = 0
            for rest in itertools.combinations(cand, t - 1):
                if tuple(sorted(rest + (x,))) in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x < 0:
            for x in range(v):
                if x not in cand_set:
                    best_x = x
                    break
        cand.append(best_x)
        cand_set.add(best_x)
    cand.sort()
    return cand


def construct(v, k, t):
    start = time.time()
    DEADLINE = start + 145.0  # safety margin before 150s kill
    rng = random.Random(12345)

    # Phase 1: greedy covering
    blocks = _greedy_cover(v, k, t, rng)
    uncovered = _count_uncovered(blocks, v, t)
    best_blocks = list(blocks)
    best_uncovered_count = len(uncovered)

    if not uncovered:
        return best_blocks

    # Phase 2: iterative local search
    # Each iteration: pick a random block, try to replace it with a block that
    # covers more uncovered t-subsets. Keep if uncovered count decreases.
    n_iter = 0
    while time.time() < DEADLINE:
        n_iter += 1
        if not uncovered:
            break
        # Pick a random block index
        i = rng.randrange(len(blocks))
        old_block = blocks[i]

        # Compute uncovered if we remove old_block
        # (i.e., t-subsets of old_block that are NOT covered by other blocks)
        covered_by_others = set()
        for j, b in enumerate(blocks):
            if j == i:
                continue
            for sub in itertools.combinations(b, t):
                covered_by_others.add(sub)
        # Uncovered after removing old_block = current uncovered + (old_block's t-subsets not in covered_by_others)
        # But we want to find a replacement that covers as many of the currently uncovered as possible.
        # Simpler: find a block that maximizes coverage of current `uncovered`.
        new_block = _best_replacement_block(old_block, uncovered, v, k, t, rng)
        if new_block is None:
            break
        if new_block == old_block:
            continue

        # Compute new uncovered if we replace old_block with new_block
        # New uncovered = (uncovered - t-subsets of new_block) + (t-subsets of old_block not covered by others)
        new_block_subs = set(itertools.combinations(new_block, t))
        old_block_subs = set(itertools.combinations(old_block, t))

        # t-subsets of old_block that become uncovered if we remove it
        # (i.e., not covered by other blocks)
        newly_uncovered_from_old = old_block_subs - covered_by_others
        # t-subsets of new_block that are currently uncovered (will be covered)
        newly_covered = new_block_subs & uncovered

        delta = len(newly_covered) - len(newly_uncovered_from_old)
        if delta < 0:
            # Improvement: uncovered count decreases
            # Update uncovered set
            uncovered.difference_update(newly_covered)
            uncovered.update(newly_uncovered_from_old)
            blocks[i] = new_block
            new_count = len(uncovered)
            if new_count < best_uncovered_count:
                best_uncovered_count = new_count
                best_blocks = list(blocks)
        # If delta >= 0, skip (no improvement); we could add simulated annealing but
        # for simplicity and speed, only accept improvements.

    return best_blocks
# EVOLVE-BLOCK-END
