"""Probabilistic weighted greedy for covering design C(24,7,3)."""
import itertools
import random
import time

# EVOLVE-BLOCK-START

def _precompute(v, k, t):
    """Precompute all t-sets and the list of t-sets containing each (t-1)-set."""
    all_tsets = list(itertools.combinations(range(v), t))
    # Map each (t-1)-tuple to the list of t-sets that contain it
    tset_by_tm1 = {}
    for ts in all_tsets:
        for i in range(t):
            tm1 = ts[:i] + ts[i + 1:]
            if tm1 not in tset_by_tm1:
                tset_by_tm1[tm1] = []
            tset_by_tm1[tm1].append(ts)
    return all_tsets, tset_by_tm1


def _weighted_greedy(v, k, t, rng, time_limit, tset_by_tm1):
    """
    Probabilistic greedy with randomized point weights.
    Each step selects the point maximizing: gain * (1 + weight) + noise,
    where weight is a per-point random value in [0,1] resampled per block.
    This escapes deterministic greedy plateaus.
    """
    deadline = time.time() + time_limit
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []

    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set to seed the block (diversity)
        first_uncovered = rng.choice(list(uncovered))
        block = list(first_uncovered)
        block_set = set(block)

        # Random weights for this block construction
        weights = {x: rng.random() for x in range(v)}

        while len(block) < k:
            best_x = None
            best_score = -1.0

            for x in range(v):
                if x in block_set:
                    continue

                # Compute gain: number of uncovered t-sets containing x
                # and (t-1) elements from current block
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    tset = tuple(sorted(S + (x,)))
                    if tset in uncovered:
                        gain += 1

                # Weighted score: gain boosted by point weight
                score = gain * (1.0 + weights[x]) + rng.random() * 0.1

                if score > best_score:
                    best_score = score
                    best_x = x

            if best_x is None:
                break
            block.append(best_x)
            block_set.add(best_x)

        if len(block) < k:
            break

        block.sort()
        blocks.append(tuple(block))

        # Remove covered t-sets
        for ts in itertools.combinations(block, t):
            uncovered.discard(tuple(ts))

    return [list(b) for b in blocks]


def _prune_redundant(blocks, v, k, t, time_limit):
    """
    Remove blocks that are redundant: a block is redundant if every t-set
    it contains is also contained in at least one other block.
    Iteratively remove until no more redundant blocks exist.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]

    def build_index(blocks):
        """Map each t-set to the list of block indices containing it."""
        idx = {}
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                idx.setdefault(tuple(ts), []).append(i)
        return idx

    changed = True
    while changed and time.time() < deadline:
        changed = False
        idx = build_index(blocks)
        to_remove = []
        for i, b in enumerate(blocks):
            redundant = True
            for ts in itertools.combinations(b, t):
                covering = idx.get(tuple(ts), [])
                if len(covering) == 1 and covering[0] == i:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)

        if to_remove:
            # Remove in reverse order to preserve indices
            for i in sorted(to_remove, reverse=True):
                blocks.pop(i)
            changed = True

    return blocks


def _local_swap(blocks, v, k, t, rng, time_limit):
    """
    Local search: for each block, check if it's redundant (remove it).
    Otherwise, try to find a replacement block that covers all critical
    t-sets and adds extra coverage, enabling future removals.
    """
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]

    def build_index(blocks):
        idx = {}
        for i, b in enumerate(blocks):
            for ts in itertools.combinations(b, t):
                idx.setdefault(tuple(ts), []).append(i)
        return idx

    while time.time() < deadline:
        idx = build_index(blocks)
        improved = False

        # Pass 1: remove redundant blocks
        to_remove = []
        for i, b in enumerate(blocks):
            redundant = True
            for ts in itertools.combinations(b, t):
                covering = idx.get(tuple(ts), [])
                if len(covering) == 1 and covering[0] == i:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)

        if to_remove:
            for i in sorted(to_remove, reverse=True):
                blocks.pop(i)
            improved = True
            continue

        # Pass 2: try to improve a random block
        if len(blocks) > 0:
            i = rng.randint(0, len(blocks) - 1)
            old_block = blocks[i]
            old_set = set(old_block)

            # Find critical t-sets (only covered by this block)
            critical = []
            for ts in itertools.combinations(old_block, t):
                covering = idx.get(tuple(ts), [])
                if len(covering) == 1 and covering[0] == i:
                    critical.append(ts)

            if not critical:
                # Shouldn't happen after pass 1, but just in case
                blocks.pop(i)
                improved = True
                continue

            # Try random replacements
            best_cand = None
            best_extra = 0
            for _ in range(30):
                if time.time() >= deadline:
                    break
                cand = sorted(rng.sample(range(v), k))
                cand_set = set(cand)

                # Must cover all critical t-sets
                covers_all = True
                for ts in critical:
                    if not set(ts).issubset(cand_set):
                        covers_all = False
                        break
                if not covers_all:
                    continue

                # Count extra coverage (t-sets not covered by any other block)
                extra = 0
                for ts in itertools.combinations(cand, t):
                    covering = idx.get(tuple(ts), [])
                    covering_without = [j for j in covering if j != i]
                    if not covering_without:
                        extra += 1

                if extra > best_extra:
                    best_extra = extra
                    best_cand = cand

            if best_cand is not None and best_extra > 0:
                blocks[i] = best_cand
                improved = True

        if not improved:
            break

    return blocks


def construct(v, k, t):
    start_time = time.time()
    total_budget = 135.0  # Safety margin for 150s kill

    # Precompute
    all_tsets, tset_by_tm1 = _precompute(v, k, t)

    best_blocks = None
    best_len = float('inf')

    # Phase 1: Multiple weighted greedy runs with different seeds
    # Each run uses a different random seed for point weights
    num_greedy = 12
    per_greedy_time = (total_budget * 0.55) / num_greedy

    for run in range(num_greedy):
        elapsed = time.time() - start_time
        remaining_for_greedy = total_budget * 0.55 - elapsed
        if remaining_for_greedy < 2.0:
            break

        rng = random.Random(42 + run * 997)
        blocks = _weighted_greedy(v, k, t, rng, per_greedy_time, tset_by_tm1)

        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]

    if best_blocks is None:
        # Fallback: simple greedy
        rng = random.Random(0)
        blocks = _weighted_greedy(v, k, t, rng, 5.0, tset_by_tm1)
        best_blocks = blocks
        best_len = len(blocks)

    # Phase 2: Prune redundant blocks
    elapsed = time.time() - start_time
    prune_time = min(10.0, total_budget * 0.15)
    if elapsed + prune_time < total_budget - 20.0:
        pruned = _prune_redundant(best_blocks, v, k, t, prune_time)
        if len(pruned) < best_len:
            best_blocks = pruned
            best_len = len(best_blocks)

    # Phase 3: Local swap search
    elapsed = time.time() - start_time
    ls_time = total_budget - elapsed - 15.0
    if ls_time > 5.0:
        rng = random.Random(777)
        refined = _local_swap(best_blocks, v, k, t, rng, ls_time)
        if len(refined) < best_len:
            best_blocks = refined
            best_len = len(best_blocks)

    # Final safety: ensure we return before deadline
    if time.time() - start_time > total_budget - 5.0:
        return best_blocks

    return best_blocks

# EVOLVE-BLOCK-END
