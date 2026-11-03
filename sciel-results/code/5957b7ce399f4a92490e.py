# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np
from math import comb

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a fast greedy + local search approach.
    Returns a list of blocks (each a list of k distinct integers in 0..v-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {frozenset(c): i for i, c in enumerate(all_tsubs)}
    idx_to_tsub = all_tsubs

    # Precompute for each point x, the t-subset indices containing x
    # This is used for fast gain computation
    # point_cover[x] = set/array of t-subset indices that contain x
    point_tsub_indices = []
    for x in range(v):
        idxs = []
        for i, c in enumerate(all_tsubs):
            if x in c:
                idxs.append(i)
        point_tsub_indices.append(np.array(idxs, dtype=np.int64))

    # Precompute (t-1) subsets for each possible (t-1)-subset as a frozenset -> list of t-subsets containing it
    # Actually, for gain: adding point x to block B covers t-subsets {x} + S where S is (t-1)-subset of B
    # So for each x, we need: for each (t-1)-subset S, the index of {x} + S (if valid)
    # Precompute: x_tsub_lookup[x] = dict mapping frozenset((t-1)-subset) -> t-subset index of {x}+subset
    x_tsub_lookup = []
    for x in range(v):
        lookup = {}
        for i, c in enumerate(all_tsubs):
            if x in c:
                rest = tuple(sorted(set(c) - {x}))
                lookup[frozenset(rest)] = i
        x_tsub_lookup.append(lookup)

    def _greedy(v, k, t, rng, deadline):
        """Fast greedy covering with vectorized gain computation."""
        uncovered = np.ones(n_tsubs, dtype=bool)
        blocks = []
        t_minus_1 = t - 1

        while uncovered.any():
            if time.time() > deadline:
                break

            # Pick a random uncovered t-subset to seed the block
            uncovered_indices = np.where(uncovered)[0]
            seed_idx = int(rng.choice(uncovered_indices))
            seed_tsub = idx_to_tsub[seed_idx]
            block = list(seed_tsub)
            block_set = set(block)

            # Build the (t-1) subsets of the current block incrementally
            # Start with empty block's (t-1) subsets = empty list (since block has t elements, 
            # actually we need (t-1) subsets of the block)
            # When block has m elements, (t-1) subsets are combinations(block, t-1)
            # We recompute each iteration for simplicity, but optimize by tracking

            while len(block) < k:
                if time.time() > deadline:
                    break

                # Compute (t-1) subsets of current block
                if len(block) >= t_minus_1:
                    t1_subsets = list(itertools.combinations(block, t_minus_1))
                else:
                    t1_subsets = []

                # For each candidate point x not in block, compute gain
                gains = np.zeros(v, dtype=np.int64)

                for x in range(v):
                    if x in block_set:
                        continue
                    if not t1_subsets:
                        continue
                    gain = 0
                    lookup = x_tsub_lookup[x]
                    for rest in t1_subsets:
                        key = frozenset(rest)
                        if key in lookup:
                            idx = lookup[key]
                            if uncovered[idx]:
                                gain += 1
                    gains[x] = gain

                # Find best point
                in_block_mask = np.zeros(v, dtype=bool)
                for x in block:
                    in_block_mask[x] = True
                valid = ~in_block_mask
                valid_gains = gains[valid]
                if len(valid_gains) > 0:
                    max_gain = int(np.max(valid_gains))
                    if max_gain > 0:
                        candidates = np.where(valid & (gains == max_gain))[0]
                        best_point = int(rng.choice(candidates))
                    else:
                        # All gains are 0, pick random
                        candidates = np.where(valid)[0]
                        best_point = int(rng.choice(candidates))
                else:
                    break

                if best_point is not None:
                    block.append(best_point)
                    block_set.add(best_point)
                else:
                    break

            if len(block) == k:
                block.sort()
                blocks.append(tuple(block))
                # Update uncovered
                for c in itertools.combinations(block, t):
                    idx = tsub_to_idx[frozenset(c)]
                    uncovered[idx] = False

        return [list(b) for b in blocks]

    def _remove_redundant(blocks, deadline):
        """Remove blocks that don't cover any unique t-subset."""
        if not blocks:
            return blocks

        # Compute coverage count for each t-subset
        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[frozenset(c)]] = True
            block_masks.append(mask)
            cov_count += mask

        # Iteratively remove redundant blocks
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(blocks)):
                mask = block_masks[i]
                # Check if block i covers any t-subset with count == 1
                if not np.any(cov_count[mask] == 1):
                    # Redundant
                    cov_count -= mask
                    blocks.pop(i)
                    block_masks.pop(i)
                    changed = True
                    break

        return blocks

    def _local_search(blocks, deadline, rng):
        """Try to improve by replacing blocks or removing redundant ones."""
        blocks = [list(b) for b in blocks]
        if not blocks:
            return blocks

        # First, remove redundant blocks
        blocks = _remove_redundant(blocks, deadline)
        if not blocks:
            return blocks

        # Compute coverage
        cov_count = np.zeros(n_tsubs, dtype=np.int64)
        block_masks = []
        for b in blocks:
            mask = np.zeros(n_tsubs, dtype=bool)
            for c in itertools.combinations(b, t):
                mask[tsub_to_idx[frozenset(c)]] = True
            block_masks.append(mask)
            cov_count += mask

        # Try single-block replacements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break

                # Find t-subsets only covered by block i
                mask_i = block_masks[i]
                unique = mask_i & (cov_count == 1)
                if not unique.any():
                    # Block i is redundant (should have been caught, but double-check)
                    cov_count -= mask_i
                    blocks.pop(i)
                    block_masks.pop(i)
                    improved = True
                    break
                    continue

                unique_indices = np.where(unique)[0]

                # Try to find a replacement block that covers all unique t-subsets
                # Strategy: the unique t-subsets must all be contained in the new block
                # So the new block must contain the union of all unique t-subsets
                # If the union has more than k elements, no single block can replace it
                unique_tsubs = [idx_to_tsub[idx] for idx in unique_indices]
                union_set = set()
                for ts in unique_tsubs:
                    union_set.update(ts)

                if len(union_set) > k:
                    continue  # Cannot replace with a single block

                # The new block must contain union_set and can have k - len(union_set) additional points
                # Try a few random completions
                remaining_points = [x for x in range(v) if x not in union_set]
                if len(remaining_points) < k - len(union_set):
                    continue

                for attempt in range(50):
                    if time.time() > deadline:
                        break
                    if len(remaining_points) <= k - len(union_set):
                        extra = remaining_points
                    else:
                        extra = rng.sample(remaining_points, k - len(union_set))
                    new_block = sorted(list(union_set) + extra)
                    new_block_set = set(new_block)

                    # Check if this new block covers all unique t-subsets
                    # (It should, since it contains the union, but verify)
                    new_mask = np.zeros(n_tsubs, dtype=bool)
                    for c in itertools.combinations(new_block, t):
                        new_mask[tsub_to_idx[frozenset(c)]] = True

                    if np.all(new_mask[unique_indices]):
                        # Check if this replacement doesn't break coverage
                        # Remove old block, add new block
                        old_cov = cov_count - mask_i
                        new_cov = old_cov + new_mask
                        if np.all(new_cov >= 1):
                            # Valid replacement
                            cov_count = new_cov
                            blocks[i] = new_block
                            block_masks[i] = new_mask
                            improved = True
                            break

        return blocks

    # Main: run greedy with multiple restarts, then local search
    best_blocks = None
    best_len = float('inf')

    # First greedy run
    rng0 = random.Random(42)
    t0 = time.time()
    baseline = _greedy(v, k, t, rng0, deadline)
    if baseline:
        best_blocks = baseline
        best_len = len(baseline)

    # More greedy restarts
    restart = 0
    while time.time() < deadline - 30:
        restart += 1
        rng = random.Random(restart * 7919 + 3)
        blocks = _greedy(v, k, t, rng, deadline)
        if blocks and len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)

    # Local search on best
    if best_blocks and time.time() < deadline - 5:
        rng_ls = random.Random(99999)
        best_blocks = _local_search(best_blocks, deadline, rng_ls)

    return best_blocks if best_blocks else []

# EVOLVE-BLOCK-END
