"""
Optimized covering design C(24,7,3) solver.

Strategy:
1. Fast greedy construction using bitmasks for t-subset tracking.
2. Local search: try to replace one block with another that covers more uncovered
   t-subsets (or equivalently, try to find a block that is more "efficient").
3. Deletion heuristic: remove blocks that are redundant (all their t-subsets are
   covered by other blocks).
4. Time-bounded: returns best valid covering found before deadline.

Key optimization: use integer bitmasks to represent sets of t-subsets.
For v=24, k=7, t=3: C(24,3)=2024 t-subsets. We can use a Python int as a
bitset of 2024 bits.
"""

import time
import itertools
import random

# Deadline: leave 10s safety margin from 120s
DEADLINE = 110.0


def _precompute_tsubsets(v, t):
    """Precompute all t-subsets as tuples and map them to indices."""
    tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(tsubs)}
    return tsubs, tsub_to_idx


def _block_mask(block, t, tsub_to_idx):
    """Return a bitmask (Python int) representing all t-subsets of block."""
    mask = 0
    for ts in itertools.combinations(block, t):
        idx = tsub_to_idx.get(ts)
        if idx is not None:
            mask |= (1 << idx)
    return mask


def _greedy_construct(v, k, t, tsubs, tsub_to_idx, rng, time_limit):
    """Greedy construction with bitmask tracking."""
    n_tsubs = len(tsubs)
    # uncovered bitmask: bit i set means tsubs[i] is uncovered
    uncovered = (1 << n_tsubs) - 1
    blocks = []
    start_time = time.time()

    # Precompute for each point x, and for each (t-1)-subset of a partial block,
    # which t-subsets would be covered. This is complex, so we use a simpler
    # approach: for each candidate point, compute how many new t-subsets it covers.

    while uncovered:
        if time.time() - start_time > time_limit:
            break

        # Find the most constrained uncovered t-subset (one that appears in fewest
        # possible blocks) -- for simplicity, just pick the first uncovered one
        # But better: pick an uncovered t-subset that maximizes the number of
        # points that can extend it. For speed, just pick first.
        # Actually, let's pick the uncovered t-subset with the highest index
        # that is uncovered -- no, let's use a smarter heuristic.

        # Get list of uncovered t-subsets
        # For speed, find the lowest set bit
        if uncovered == 0:
            break

        # Pick an uncovered t-subset to seed the block
        # Use a heuristic: pick the uncovered t-subset that has the most
        # "flexibility" -- actually for speed just pick first uncovered
        # Find first set bit
        # In Python, we can do:
        lowbit = uncovered & (-uncovered)
        seed_idx = lowbit.bit_length() - 1
        seed = tsubs[seed_idx]

        block = list(seed)
        block_set = set(seed)

        # Build the block greedily
        while len(block) < k:
            if time.time() - start_time > time_limit:
                break

            # For each candidate point, compute gain
            best_x = -1
            best_gain = -1

            # Precompute: for each (t-1)-subset of current block, which points
            # would complete it to a t-subset that is uncovered
            # For each candidate x, gain = number of (t-1)-subsets S of block
            # such that S ∪ {x} is an uncovered t-subset

            # Build a dict: for each (t-1)-subset, count uncovered t-subsets
            # that contain it and a specific point
            # Actually simpler: for each x not in block, count how many
            # (t-1)-subsets S of block have S∪{x} uncovered

            # Precompute the (t-1)-subsets of current block
            partial_subs = list(itertools.combinations(block, t - 1))

            # For each partial sub, precompute which points would make it uncovered
            # gain[x] = sum over partial_subs S: [S ∪ {x} is uncovered]

            # Build a lookup: for each partial sub S (as tuple), and point x,
            # check if S∪{x} is in uncovered
            # To speed up, precompute for each partial sub S, the set of points
            # x such that S∪{x} is uncovered

            # For each partial sub, the possible completions are points not in S
            # We need to check which of those completions give an uncovered t-subset

            # Let's just iterate over candidates
            candidates = [x for x in range(v) if x not in block_set]

            # For speed, precompute for each partial sub a bitmask of points
            # that would complete it to an uncovered t-subset
            # But this is complex. Let's just do direct counting.

            # Optimize: for each candidate x, gain = 0
            # For each partial sub S of block:
            #   ts = tuple(sorted(S + (x,)))
            #   if ts in tsub_to_idx and (uncovered >> tsub_to_idx[ts]) & 1:
            #       gain[x] += 1

            # Precompute tsub_to_idx lookup for speed
            # This is O(|block|^{t-1} * (v - |block|)) per iteration

            gains = {}
            for x in candidates:
                g = 0
                for S in partial_subs:
                    # S is a (t-1)-tuple, x not in S (since x not in block)
                    ts = tuple(sorted(S + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None:
                        if (uncovered >> idx) & 1:
                            g += 1
                gains[x] = g

            if best_gain < 0:
                # Pick the candidate with max gain
                for x in candidates:
                    if gains[x] > best_gain:
                        best_gain = gains[x]
                        best_x = x
                if best_x == -1:
                    # No gain, pick first candidate
                    best_x = candidates[0] if candidates else 0

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered
        bmask = _block_mask(block, t, tsub_to_idx)
        uncovered &= ~bmask

    return blocks


def _remove_redundant(blocks, v, k, t, tsub_to_idx):
    """Remove blocks whose t-subsets are all covered by other blocks."""
    if not blocks:
        return blocks

    n_blocks = len(blocks)
    # For each block, compute its mask
    masks = [_block_mask(b, t, tsub_to_idx) for b in blocks]

    # For each block i, check if mask[i] is a subset of OR of all other masks
    # i.e., mask[i] & ~(OR of others) == 0
    # OR of all masks:
    total_or = 0
    for m in masks:
        total_or |= m

    kept = []
    for i in range(n_blocks):
        others_or = total_or & ~masks[i]
        # Actually we need OR of all masks except i
        # total_or includes mask[i], so others_or = total_or with mask[i] bits
        # that are only in mask[i] removed. But if a bit is in mask[i] and also
        # in some other mask, it's still in total_or.
        # The check: is mask[i] ⊆ (OR of masks[j] for j≠i)?
        # This is equivalent to: mask[i] & ~(OR of masks[j] for j≠i) == 0
        # OR of masks[j] for j≠i is not simply total_or & ~mask[i] because
        # total_or has bits from mask[i] too.
        # Correct approach: compute OR of all masks except i
        # For efficiency, we can precompute prefix and suffix ORs, but for
        # small n_blocks this is fine.

        # For now, compute directly (slow but correct)
        others = 0
        for j in range(n_blocks):
            if j != i:
                others |= masks[j]

        if masks[i] & ~others == 0:
            # Block i is redundant
            continue
        else:
            kept.append(blocks[i])

    return kept


def _local_search(blocks, v, k, t, tsubs, tsub_to_idx, rng, time_limit):
    """
    Local search: try to improve the covering by replacing blocks.
    Strategy: for each block, try replacing it with a random block that
    covers more uncovered t-subsets.
    """
    start_time = time.time()
    n_tsubs = len(tsubs)

    def get_uncovered(blocks):
        covered = 0
        for b in blocks:
            covered |= _block_mask(b, t, tsub_to_idx)
        return ~covered & ((1 << n_tsubs) - 1)

    uncovered = get_uncovered(blocks)

    while time.time() - start_time < time_limit:
        if not uncovered:
            break

        # Try to find a block replacement that reduces uncovered
        improved = False
        n_blocks = len(blocks)

        for i in range(n_blocks):
            if time.time() - start_time > time_limit:
                break

            old_block = blocks[i]
            old_mask = _block_mask(old_block, t, tsub_to_idx)

            # Try a few random replacements
            for attempt in range(5):
                if time.time() - start_time > time_limit:
                    break

                # Generate a random block that includes some points from old_block
                # to maintain coverage
                # Pick 3-4 points from old_block, then random others
                n_keep = min(len(old_block), rng.randint(3, 5))
                keep = rng.sample(old_block, n_keep)
                remaining = [x for x in range(v) if x not in keep]
                if len(remaining) < k - n_keep:
                    continue
                fill = rng.sample(remaining, k - n_keep)
                new_block = tuple(sorted(keep + fill))

                new_mask = _block_mask(new_block, t, tsub_to_idx)

                # Check if new block covers more uncovered t-subsets than old
                # Uncovered covered by old: old_mask & uncovered
                # Uncovered covered by new: new_mask & uncovered
                old_gain = (old_mask & uncovered).bit_count()
                new_gain = (new_mask & uncovered).bit_count()

                if new_gain > old_gain:
                    blocks[i] = list(new_block)
                    # Update uncovered
                    # Recompute is expensive; instead:
                    # uncovered changes: bits in old_mask that were uncovered and
                    # not in new_mask become uncovered; bits in new_mask that were
                    # uncovered and not in old_block become covered
                    # Actually, let's just recompute for correctness
                    uncovered = get_uncovered(blocks)
                    improved = True
                    break

        if not improved:
            # No improvement in this pass, try deletion
            new_blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx)
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                uncovered = get_uncovered(blocks)
            else:
                # Stuck, break
                break

    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v, k, t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    rng = random.Random(42)

    tsubs, tsub_to_idx = _precompute_tsubsets(v, t)
    n_tsubs = len(tsubs)

    # Phase 1: Greedy construction
    greedy_time = min(30.0, (deadline - start_time) * 0.4)
    blocks = _greedy_construct(v, k, t, tsubs, tsub_to_idx, rng, greedy_time)

    # Check if we have a valid covering
    if not blocks:
        # Fallback: brute force (shouldn't happen)
        return [list(c) for c in itertools.combinations(range(v), k)]

    # Phase 2: Remove redundant blocks
    if time.time() < deadline - 5:
        blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx)

    # Phase 3: Local search
    ls_time = min(60.0, max(5.0, deadline - time.time() - 5))
    if ls_time > 5:
        blocks = _local_search(blocks, v, k, t, tsubs, tsub_to_idx, rng, ls_time)

    # Phase 4: Final deletion pass
    if time.time() < deadline - 2:
        blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx)

    # Verify validity
    covered = 0
    for b in blocks:
        covered |= _block_mask(b, t, tsub_to_idx)
    if covered != (1 << n_tsubs) - 1:
        # Not a valid covering; fall back to greedy with more time
        # Try again with more aggressive greedy
        remaining_time = deadline - time.time() - 2
        if remaining_time > 10:
            blocks2 = _greedy_construct(v, k, t, tsubs, tsub_to_idx, rng, remaining_time)
            # Merge: use the smaller valid one
            covered2 = 0
            for b in blocks2:
                covered2 |= _block_mask(b, t, tsub_to_idx)
            if covered2 == (1 << n_tsubs) - 1:
                blocks = blocks2
                blocks = _remove_redundant(blocks, v, k, t, tsub_to_idx)

    # Ensure blocks are sorted and distinct
    blocks = [sorted(b) for b in blocks]
    seen = set()
    unique_blocks = []
    for b in blocks:
        key = tuple(b)
        if key not in seen:
            seen.add(key)
            unique_blocks.append(b)

    return unique_blocks
