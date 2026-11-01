"""Covering design C(19,8,5) - multi-start greedy with fast vectorized gain + pruning."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    # Precompute all t-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    n_tsets = len(all_tsets)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}

    # Precompute for each t-set: the bitmask of points (v=19 fits in int)
    tset_masks = np.zeros(n_tsets, dtype=np.uint32)
    for i, ts in enumerate(all_tsets):
        mask = 0
        for p in ts:
            mask |= (1 << p)
        tset_masks[i] = mask

    # Precompute: for each (t-1)-subset of points, the list of t-set indices that extend it
    # (t-1)-subset -> list of (point, tset_idx) pairs
    # Actually: for gain computation, when we have block B and consider adding x,
    # gain = number of uncovered tsets ts such that ts \ {x} ⊆ B and x ∈ ts
    # = number of (t-1)-subsets S of B such that S ∪ {x} is a t-set and that t-set is uncovered
    # Precompute: for each (t-1)-tuple, map to tset index
    t_minus_1_to_tset = {}  # frozenset of (t-1) points -> list of (x, tset_idx)
    for i, ts in enumerate(all_tsets):
        ts_set = set(ts)
        for x in ts:
            rest = frozenset(ts_set - {x})
            key = tuple(sorted(rest))
            if key not in t_minus_1_to_tset:
                t_minus_1_to_tset[key] = []
            t_minus_1_to_tset[key].append((x, i))

    def _greedy_single(rng, deadline_local):
        """Single greedy run: repeatedly pick best block covering most uncovered t-sets."""
        uncovered = np.ones(n_tsets, dtype=bool)
        num_uncovered = n_tsets
        blocks = []

        while num_uncovered > 0 and time.time() < deadline_local:
            # Pick a random uncovered t-set as anchor
            uncov_indices = np.nonzero(uncovered)[0]
            if len(uncov_indices) == 0:
                break
            seed_idx = int(uncov_indices[rng.randrange(len(uncov_indices))])
            seed = all_tsets[seed_idx]

            # Build block starting from seed
            block = list(seed)
            block_set = set(block)
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)

            # Greedily add points to maximize coverage of uncovered t-sets
            while len(block) < k and time.time() < deadline_local:
                best_x = -1
                best_gain = -1

                # Current block covers t-sets where tset_mask ⊆ block_mask
                # Adding x: new t-sets covered are those with (tset_mask & ~block_mask) == (1<<x)
                # i.e., tset_mask has exactly one bit outside block, and that bit is x
                # And the tset must be uncovered

                # Fast vectorized check:
                # For each candidate x, compute which uncovered t-sets would be newly covered
                # A tset i is newly covered by x iff:
                #   (tset_masks[i] & ~block_mask) == (1 << x)  AND  uncovered[i]
                #   AND  (tset_masks[i] & block_mask) == (tset_masks[i] & ~(1<<x))
                # Simplify: tset_masks[i] | block_mask == tset_masks[i] | (block_mask | (1<<x))
                # and tset_masks[i] & ~(block_mask) == (1<<x)

                remaining_mask = np.zeros(n_tsets, dtype=np.uint32)
                for i in range(n_tsets):
                    remaining_mask[i] = tset_masks[i] & ~block_mask

                # For each x, count uncovered tsets where remaining_mask[i] == (1<<x)
                # and tset_masks[i] has x set (implied by remaining_mask == 1<<x)
                # Actually remaining_mask[i] == (1<<x) means exactly one point outside block, which is x
                # AND that tset must contain all points in block: tset_masks[i] & block_mask == block_mask... 
                # No: remaining_mask = tset & ~block. If remaining_mask == (1<<x), then tset = (tset & block) | (1<<x)
                # This means all non-x points of tset are in block. Good.
                # But we also need tset to contain x, which is guaranteed since x is in remaining.
                # We also need the tset to be a valid subset of block ∪ {x}, which is guaranteed.

                for x in range(v):
                    if x in block_set:
                        continue
                    x_mask = np.uint32(1 << x)
                    # Count uncovered tsets where remaining_mask == x_mask
                    # But we also need tset to actually contain all block points? 
                    # remaining_mask == x_mask means tset \ block = {x}, so tset ⊆ block ∪ {x} and x ∈ tset.
                    # This is exactly what we want.
                    mask_match = (remaining_mask == x_mask)
                    gain = int(np.sum(mask_match & uncovered))
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x

                if best_x == -1 or best_gain <= 0:
                    break
                block.append(best_x)
                block_set.add(best_x)
                block_mask |= (1 << best_x)

            block.sort()
            blocks.append(block)

            # Mark covered t-sets
            # A tset is covered if tset_mask ⊆ block_mask
            # i.e., tset_masks[i] & ~block_mask == 0
            newly_covered = (tset_masks & ~np.uint32(block_mask)) == 0
            newly_covered &= uncovered
            num_new = int(np.sum(newly_covered))
            if num_new > 0:
                uncovered[newly_covered] = False
                num_uncovered -= num_new

        return blocks

    def _prune_blocks(blocks, deadline_local):
        """Remove redundant blocks iteratively."""
        blocks = [list(b) for b in blocks]
        if not blocks:
            return blocks

        # coverage_count per tset
        coverage_count = np.zeros(n_tsets, dtype=np.int32)
        # For each block, list of tset indices it covers
        block_tset_lists = []
        for block in blocks:
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)
            covered = np.nonzero((tset_masks & ~np.uint32(block_mask)) == 0)[0]
            block_tset_lists.append(covered)
            coverage_count[covered] += 1

        improved = True
        while improved and time.time() < deadline_local:
            improved = False
            # Find block with minimum unique coverage
            best_remove = -1
            best_unique = n_tsets + 1
            for bi in range(len(blocks)):
                if time.time() >= deadline_local:
                    break
                covered = block_tset_lists[bi]
                if len(covered) == 0:
                    best_remove = bi
                    best_unique = 0
                    break
                unique = int(np.sum(coverage_count[covered] == 1))
                if unique < best_unique:
                    best_unique = unique
                    best_remove = bi
                if unique == 0:
                    break

            if best_remove == -1 or best_unique > 0:
                # Only remove if no unique coverage
                if best_remove != -1 and best_unique == 0:
                    covered = block_tset_lists[best_remove]
                    coverage_count[covered] -= 1
                    blocks.pop(best_remove)
                    block_tset_lists.pop(best_remove)
                    improved = True
                else:
                    break
            else:
                covered = block_tset_lists[best_remove]
                coverage_count[covered] -= 1
                blocks.pop(best_remove)
                block_tset_lists.pop(best_remove)
                improved = True

        return blocks

    def _verify(blocks):
        """Check all t-sets are covered."""
        covered = np.zeros(n_tsets, dtype=bool)
        for block in blocks:
            block_mask = 0
            for p in block:
                block_mask |= (1 << p)
            newly = (tset_masks & ~np.uint32(block_mask)) == 0
            covered |= newly
        return bool(np.all(covered))

    # Multi-start greedy
    rng = random.Random(42)
    best_blocks = None
    best_count = float('inf')

    # Allocate time: ~8s per greedy run, leave 15s for pruning
    greedy_deadline = deadline - 20
    num_runs = 0

    while time.time() < greedy_deadline and num_runs < 20:
        # Give each greedy run about 7 seconds
        run_deadline = min(time.time() + 7, greedy_deadline)
        if run_deadline <= time.time() + 1:
            break
        blocks = _greedy_single(rng, run_deadline)
        if blocks and len(blocks) < best_count:
            # Quick verify
            if _verify(blocks):
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
        num_runs += 1

    if best_blocks is None:
        # Fallback: one more try with full time
        blocks = _greedy_single(rng, deadline - 5)
        best_blocks = blocks
        best_count = len(blocks)

    # Prune
    if time.time() < deadline - 3:
        best_blocks = _prune_blocks(best_blocks, deadline - 1)
        best_count = len(best_blocks)

    return best_blocks
