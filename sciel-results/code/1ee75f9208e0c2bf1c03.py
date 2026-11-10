"""
Covering design C(19,8,5) – strong greedy with bitset acceleration,
randomized restarts, and a redundant-block removal pass.
"""
import time
import random
import itertools
import numpy as np
from collections import defaultdict

# ── precomputation ────────────────────────────────────────────────────────────

def _precompute(v, k, t):
    """
    Build all data structures needed for fast greedy.
    Returns a dict with:
      all_tsubsets : list of all t-subsets (as tuples)
      tsub_idx     : dict  t-subset tuple -> int index
      tsub_block_mask : for each t-subset, a set of k-subsets that contain it
                        (not used directly; instead we use point-level masks)
      point_tsub_mask : for each point x, a numpy bool array of length n_tsub
                        where mask[i]=True iff t-subset i contains x
    """
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsubsets)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubsets)}

    # point_tsub_mask[x][i] = True iff x is in all_tsubsets[i]
    point_tsub_mask = np.zeros((v, n_tsub), dtype=bool)
    for i, ts in enumerate(all_tsubsets):
        for x in ts:
            point_tsub_mask[x, i] = True

    return {
        'all_tsubsets': all_tsubsets,
        'tsub_idx': tsub_idx,
        'n_tsub': n_tsub,
        'point_tsub_mask': point_tsub_mask,
    }


def _count_uncovered_in_block(block, covered, tsub_idx, t):
    """
    For a candidate block (tuple of k points), return a list of indices of
    currently-uncovered t-subsets that this block would cover.
    Uses the tsub_idx dict for O(1) lookup.
    """
    newly = []
    for ts in itertools.combinations(block, t):
        i = tsub_idx.get(ts, -1)
        if i != -1 and not covered[i]:
            newly.append(i)
    return newly


def _greedy_build_block(block_seed, covered, tsub_idx, point_tsub_mask, v, k, t,
                        rng, max_iters=200):
    """
    Starting from an initial set of points (block_seed), greedily add points
    to maximize the number of newly-covered t-subsets, until the block has k
    points or no further gain is possible.
    Returns (block_tuple, list_of_newly_covered_indices).
    """
    block_set = set(block_seed)
    # current coverage mask for this partial block
    # we track which t-subsets are covered by the current block_set
    # and are currently uncovered globally
    # Instead of maintaining a running mask, we recompute gain for each candidate point.

    for _ in range(k - len(block_set)):
        best_x = None
        best_gain = -1
        best_newly = None

        for x in range(v):
            if x in block_set:
                continue
            # gain = number of currently-uncovered t-subsets that contain x
            #        and whose other t-1 elements are all in block_set
            # We compute this by checking all (t-1)-subsets of block_set
            gain = 0
            newly = []
            if len(block_set) >= t - 1:
                for rest in itertools.combinations(block_set, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    i = tsub_idx.get(ts, -1)
                    if i != -1 and not covered[i]:
                        gain += 1
                        newly.append(i)
            # Also, t-subsets that are entirely within block_set ∪ {x} and
            # contain x but not all other elements in block_set are already
            # handled above only when exactly t-1 from block_set.
            # Actually we need: t-subset S with x in S and |S ∩ block_set| >= t-1
            # Since S has size t and x in S, S\{x} has size t-1.
            # S is covered by block_set ∪ {x} iff S\{x} ⊆ block_set.
            # So we need all (t-1)-subsets of block_set that together with x
            # form a valid t-subset. That's exactly what we did above.
            # But also t-subsets where S\{x} is a (t-1)-subset of block_set
            # is the only case. Good.

            if gain > best_gain:
                best_gain = gain
                best_x = x
                best_newly = newly
            elif gain == best_gain and gain > 0 and best_x is not None:
                # random tie-break
                if rng.random() < 0.3:
                    best_x = x
                    best_newly = newly

        if best_x is None or best_gain <= 0:
            break
        block_set.add(best_x)

    block = tuple(sorted(block_set))
    newly = _count_uncovered_in_block(block, covered, tsub_idx, t)
    return block, newly


def _fast_greedy(v, k, t, data, seed=42, max_time=30.0, randomize=True):
    """
    Fast greedy covering with randomized restarts inside.
    Returns list of blocks.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time

    all_tsubsets = data['all_tsubsets']
    tsub_idx = data['tsub_idx']
    n_tsub = data['n_tsub']
    point_tsub_mask = data['point_tsub_mask']

    covered = np.zeros(n_tsub, dtype=bool)
    num_uncovered = n_tsub
    blocks = []
    seen_blocks = set()

    while num_uncovered > 0:
        if time.time() > deadline:
            break

        # Find uncovered t-subset indices
        uncovered_indices = np.where(~covered)[0]
        if len(uncovered_indices) == 0:
            break

        # Sample up to 30 uncovered t-subsets as seeds
        if randomize and len(uncovered_indices) > 30:
            sample = rng.sample(list(uncovered_indices), 30)
        else:
            sample = list(uncovered_indices[:30])

        best_block = None
        best_gain = -1
        best_newly = None

        for idx in sample:
            focus = all_tsubsets[idx]
            block, newly = _greedy_build_block(
                list(focus), covered, tsub_idx, point_tsub_mask,
                v, k, t, rng
            )
            if block in seen_blocks:
                continue
            gain = len(newly)
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_newly = newly
            elif gain == best_gain and best_block is not None and randomize:
                if rng.random() < 0.2:
                    best_block = block
                    best_newly = newly

        if best_block is None:
            # Fallback: pick a random uncovered t-subset and build around it
            idx = int(uncovered_indices[rng.randint(0, len(uncovered_indices) - 1)])
            focus = all_tsubsets[idx]
            best_block, best_newly = _greedy_build_block(
                list(focus), covered, tsub_idx, point_tsub_mask,
                v, k, t, rng
            )
            if best_block is None or len(best_newly) == 0:
                break

        seen_blocks.add(best_block)
        blocks.append(list(best_block))

        for i in best_newly:
            if not covered[i]:
                covered[i] = True
                num_uncovered -= 1

    return blocks


def _remove_redundant_blocks(blocks, v, k, t, data, max_time=15.0):
    """
    Remove blocks whose t-subsets are all covered by other blocks.
    """
    start_time = time.time()
    deadline = start_time + max_time
    tsub_idx = data['tsub_idx']
    n_tsub = data['n_tsub']

    # Count how many blocks cover each t-subset
    tsub_count = np.zeros(n_tsub, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            i = tsub_idx.get(ts, -1)
            if i != -1:
                tsub_count[i] += 1

    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsub_idx.get(ts, -1)
                if idx != -1 and tsub_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(block, t):
                    idx = tsub_idx.get(ts, -1)
                    if idx != -1:
                        tsub_count[idx] -= 1
                blocks.pop(i)
                improved = True

    return blocks


def _local_search_swap(blocks, v, k, t, data, max_time=15.0, seed=42):
    """
    Try to improve the covering by replacing one block with another that
    covers the same t-subsets but allows removing another block, or simply
    try to find a block that covers more uncovered t-subsets.
    
    For simplicity, we do a limited pass: for each block, try to find a
    replacement block that, when substituted, allows removing at least one
    other block. This is expensive, so we limit the search.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    tsub_idx = data['tsub_idx']
    n_tsub = data['n_tsub']
    all_tsubsets = data['all_tsubsets']

    # Build coverage count
    tsub_count = np.zeros(n_tsub, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            i = tsub_idx.get(ts, -1)
            if i != -1:
                tsub_count[i] += 1

    # For each block, compute its "essential" t-subsets (those with count 1)
    # If a block has no essential t-subsets, it's already redundant (handled above).
    # Try to swap: replace block i with a new block that covers all essential
    # t-subsets of block i, and check if this allows removing another block.
    # This is complex; instead, do a simpler heuristic:
    # Try to find a block that covers the most uncovered t-subsets and add it,
    # then re-run redundant removal. But this increases block count.
    
    # Alternative: for each block, try to find a "better" block that covers
    # all its t-subsets and additionally covers some t-subsets that are
    # currently only covered by one other block, allowing that other block
    # to be removed.
    
    # Given time constraints, we'll do a limited version:
    # For a random subset of blocks, try to find a replacement.
    
    n_blocks = len(blocks)
    if n_blocks == 0:
        return blocks

    # Sample up to 20 blocks to try swapping
    candidates = rng.sample(range(n_blocks), min(20, n_blocks))

    for bi in candidates:
        if time.time() > deadline:
            break
        if bi >= len(blocks):
            continue
        old_block = blocks[bi]
        old_tsubs = list(itertools.combinations(old_block, t))
        old_tsub_indices = []
        for ts in old_tsubs:
            i = tsub_idx.get(ts, -1)
            if i != -1:
                old_tsub_indices.append(i)

        # Essential t-subsets of this block
        essential = [i for i in old_tsub_indices if tsub_count[i] == 1]
        if len(essential) == 0:
            continue  # already redundant

        # Try to find a new block that covers all essential t-subsets
        # and maximizes additional coverage
        # Build a candidate: start with points from essential t-subsets
        essential_points = set()
        for i in essential:
            essential_points.update(all_tsubsets[i])

        if len(essential_points) > k:
            continue  # can't fit in one block

        # Greedily build a block around essential_points
        block_set = set(essential_points)
        # Try to complete to k points, maximizing coverage of t-subsets
        # that are currently covered only once (by other blocks)
        
        for _ in range(k - len(block_set)):
            best_x = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                # Gain: number of t-subsets S such that x in S,
                # S\{x} ⊆ block_set, and tsub_count[idx(S)] >= 1
                # (i.e., S is currently covered, and adding this block
                #  would give it an extra cover, potentially allowing
                #  removal of another block)
                gain = 0
                if len(block_set) >= t - 1:
                    for rest in itertools.combinations(block_set, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        i = tsub_idx.get(ts, -1)
                        if i != -1 and tsub_count[i] >= 1:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None or best_gain <= 0:
                break
            block_set.add(best_x)

        new_block = tuple(sorted(block_set))
        if new_block == tuple(sorted(old_block)):
            continue

        # Check if new_block covers all essential t-subsets
        new_tsubs = set(itertools.combinations(new_block, t))
        covers_essential = all(
            tuple(sorted(all_tsubsets[i])) in new_tsubs for i in essential
        )
        if not covers_essential:
            continue

        # Check if replacing old_block with new_block allows removing another block
        # Compute new tsub_count after replacement
        new_count = tsub_count.copy()
        # Remove old block's contribution
        for i in old_tsub_indices:
            new_count[i] -= 1
        # Add new block's contribution
        new_tsub_indices = []
        for ts in itertools.combinations(new_block, t):
            i = tsub_idx.get(ts, -1)
            if i != -1:
                new_count[i] += 1
                new_tsub_indices.append(i)

        # Check if any block j != bi can now be removed
        removed = False
        for j in range(len(blocks)):
            if j == bi:
                continue
            block_j = blocks[j]
            can_remove = True
            for ts in itertools.combinations(block_j, t):
                idx = tsub_idx.get(ts, -1)
                if idx != -1 and new_count[idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                removed = True
                break

        if removed:
            # Perform the swap and removal
            blocks[bi] = list(new_block)
            # Find and remove the redundant block
            for j in range(len(blocks)):
                if j == bi:
                    continue
                block_j = blocks[j]
                can_remove = True
                for ts in itertools.combinations(block_j, t):
                    idx = tsub_idx.get(ts, -1)
                    if idx != -1 and new_count[idx] <= 1:
                        can_remove = False
                        break
                if can_remove:
                    blocks.pop(j)
                    break
            # Update tsub_count
            tsub_count = new_count
            # Remove old block's contribution and add new
            # (already done in new_count)

    return blocks


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    total_start = time.time()
    total_deadline = total_start + 135.0  # safety margin before 150s kill

    data = _precompute(v, k, t)

    best_blocks = None
    best_len = float('inf')

    # Run multiple greedy strategies with different seeds
    seeds = [1, 42, 123, 777, 2024, 999, 555, 314, 271, 1618, 7, 13, 21, 34, 55]

    for i, seed in enumerate(seeds):
        if time.time() > total_deadline:
            break

        remaining = total_deadline - time.time()
        if remaining < 5.0:
            break

        # Allocate time: earlier runs get more time
        if i < 3:
            alloc = min(remaining * 0.4, 30.0)
        elif i < 8:
            alloc = min(remaining * 0.3, 20.0)
        else:
            alloc = min(remaining * 0.2, 15.0)

        blocks = _fast_greedy(v, k, t, data, seed=seed, max_time=alloc, randomize=True)

        if blocks:
            # Quick redundant removal
            blocks = _remove_redundant_blocks(blocks, v, k, t, data, max_time=5.0)
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks

    if best_blocks is None:
        # Fallback: pure greedy
        blocks = _fast_greedy(v, k, t, data, seed=0, max_time=120.0, randomize=False)
        return blocks

    # Final refinement passes
    if time.time() < total_deadline - 10.0:
        refine_time = min(total_deadline - time.time() - 10.0, 20.0)
        best_blocks = _remove_redundant_blocks(
            best_blocks, v, k, t, data, max_time=refine_time * 0.5
        )

    if time.time() < total_deadline - 5.0:
        ls_time = min(total_deadline - time.time() - 5.0, 15.0)
        best_blocks = _local_search_swap(
            best_blocks, v, k, t, data, max_time=ls_time, seed=99
        )
        best_blocks = _remove_redundant_blocks(
            best_blocks, v, k, t, data, max_time=5.0
        )

    return best_blocks
