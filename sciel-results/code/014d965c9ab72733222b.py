"""Covering design C(19,8,5) - structural rewrite with different search strategy."""
import itertools
import time
import random
import numpy as np
from collections import defaultdict


def _precompute_tsets(v, t):
    """Precompute all t-subsets and map them to indices."""
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    return all_tsets, tset_to_idx


def _greedy_cover(v, k, t, all_tsets, tset_to_idx, rng=None, deadline=None):
    """
    Greedy set-cover with vectorized gain computation.
    Uses numpy for fast gain calculation.
    """
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = time.time() + 110

    n_tsets = len(all_tsets)
    # uncovered[i] = True if tset i is still uncovered
    uncovered = np.ones(n_tsets, dtype=bool)
    num_uncovered = n_tsets
    blocks = []

    # Precompute for each t-set, which points it contains
    # tset_points[i] = list of points in tset i
    tset_points = [set(ts) for ts in all_tsets]

    while num_uncovered > 0 and time.time() < deadline:
        # Pick a random uncovered t-set as seed
        # Find an uncovered index
        uncov_indices = np.where(uncovered)[0]
        if len(uncov_indices) == 0:
            break
        seed_idx = uncov_indices[rng.randrange(len(uncov_indices))]
        seed = all_tsets[seed_idx]
        seed_set = set(seed)

        # Build block by greedily adding points
        block = list(seed)
        block_set = set(block)
        # Track which t-sets are covered by the current block
        # A t-set is covered if all its points are in block_set

        while len(block) < k and time.time() < deadline:
            # For each candidate point x not in block, compute gain
            # gain(x) = number of uncovered t-sets that contain all points of block
            #           except possibly x, and contain x
            # More precisely: t-sets that are subsets of block_set ∪ {x}
            #                and were uncovered before adding x

            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue

                # A t-set ts is newly covered by adding x iff:
                #   ts ⊆ block_set ∪ {x}  AND  x ∈ ts  AND  ts was uncovered
                #   i.e., ts \ {x} ⊆ block_set

                # We need to count uncovered t-sets where ts \ {x} ⊆ block_set and x ∈ ts
                # Equivalently: ts ⊆ block_set ∪ {x}, x ∈ ts, ts uncovered

                gain = 0
                # Iterate over uncovered t-sets - but this is expensive
                # Instead: for each (t-1)-subset of block, check if adding x covers a t-set
                if len(block) >= t - 1:
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tset_to_idx.get(ts, -1)
                        if idx >= 0 and uncovered[idx]:
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

        # Mark all t-subsets of this block as covered
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx.get(tuple(sorted(ts)), -1)
            if idx >= 0 and uncovered[idx]:
                uncovered[idx] = False
                num_uncovered -= 1

    return blocks


def _greedy_cover_v2(v, k, t, all_tsets, tset_to_idx, rng=None, deadline=None):
    """
    Faster greedy: use a different strategy - start from a random t-set,
    then use precomputed incidence to speed up gain computation.
    """
    if rng is None:
        rng = random.Random()
    if deadline is None:
        deadline = time.time() + 110

    n_tsets = len(all_tsets)
    uncovered = np.ones(n_tsets, dtype=bool)
    num_uncovered = n_tsets
    blocks = []

    # For speed: precompute for each point, the list of t-set indices containing it
    point_to_tsets = defaultdict(list)
    for i, ts in enumerate(all_tsets):
        for p in ts:
            point_to_tsets[p].append(i)

    while num_uncovered > 0 and time.time() < deadline:
        # Pick random uncovered t-set
        uncov_mask = uncovered
        # Use numpy to find an uncovered index
        uncov_indices = np.nonzero(uncov_mask)[0]
        if len(uncov_indices) == 0:
            break
        seed_idx = int(uncov_indices[rng.randrange(len(uncov_indices))])
        seed = all_tsets[seed_idx]

        block = list(seed)
        block_set = set(block)

        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1

            for x in range(v):
                if x in block_set:
                    continue
                # Gain: number of uncovered t-sets ts where ts⊆block_set∪{x} and x∈ts
                # = number of uncovered t-sets containing x and (ts\{x})⊆block_set
                gain = 0
                if len(block) >= t - 1:
                    # For each (t-1)-subset of block, the t-set is combo + (x,)
                    for combo in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(combo + (x,)))
                        idx = tset_to_idx.get(ts, -1)
                        if idx >= 0 and uncovered[idx]:
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

        for ts in itertools.combinations(block, t):
            idx = tset_to_idx.get(tuple(sorted(ts)), -1)
            if idx >= 0 and uncovered[idx]:
                uncovered[idx] = False
                num_uncovered -= 1

    return blocks


def _local_search_remove(blocks, v, k, t, all_tsets, tset_to_idx, deadline):
    """Remove redundant blocks."""
    blocks = [list(b) for b in blocks]
    n_blocks = len(blocks)

    # coverage_count[ts_idx] = number of blocks covering this t-set
    coverage_count = np.zeros(len(all_tsets), dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx.get(tuple(sorted(ts)), -1)
            if idx >= 0:
                coverage_count[idx] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        for bi in range(len(blocks)):
            if time.time() >= deadline:
                break
            # Check if block bi is redundant
            redundant = True
            for ts in itertools.combinations(blocks[bi], t):
                idx = tset_to_idx.get(tuple(sorted(ts)), -1)
                if idx >= 0 and coverage_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(blocks[bi], t):
                    idx = tset_to_idx.get(tuple(sorted(ts)), -1)
                    if idx >= 0:
                        coverage_count[idx] -= 1
                blocks.pop(bi)
                improved = True
                break

    return blocks


def _local_search_replace(blocks, v, k, t, all_tsets, tset_to_idx, deadline):
    """
    Try to replace a block with a smaller 'effective' contribution.
    Actually, try to find if removing a block and re-covering its unique t-sets
    with fewer new blocks is possible. Simplified: just try removals.
    """
    return _local_search_remove(blocks, v, k, t, all_tsets, tset_to_idx, deadline)


def _spectral_prune(blocks, v, k, t, all_tsets, tset_to_idx, deadline):
    """
    Spectral/conductance-based pruning:
    Build a bipartite graph: blocks on one side, t-sets on the other.
    Edge if block covers t-set.
    For each block, compute its 'conductance' = (number of t-sets it uniquely covers) / (total t-sets it covers).
    Blocks with low conductance are more redundant. Remove them if coverage is maintained.
    """
    blocks = [list(b) for b in blocks]
    n_blocks = len(blocks)
    n_tsets = len(all_tsets)

    # Build coverage matrix (sparse)
    # For each block, list of t-set indices it covers
    block_tsets = []
    for block in blocks:
        tsets_covered = []
        for ts in itertools.combinations(block, t):
            idx = tset_to_idx.get(tuple(sorted(ts)), -1)
            if idx >= 0:
                tsets_covered.append(idx)
        block_tsets.append(tsets_covered)

    # Coverage count
    coverage_count = np.zeros(n_tsets, dtype=np.int32)
    for tsets in block_tsets:
        for idx in tsets:
            coverage_count[idx] += 1

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Compute conductance for each block
        conductances = []
        for i, tsets in enumerate(block_tsets):
            if len(tsets) == 0:
                conductances.append(0.0)
                continue
            unique = sum(1 for idx in tsets if coverage_count[idx] == 1)
            conductances.append(unique / len(tsets))

        # Try removing the block with lowest conductance
        if not blocks:
            break
        min_idx = int(np.argmin(conductances))
        min_cond = conductances[min_idx]

        # Check if removal is safe
        safe = True
        for idx in block_tsets[min_idx]:
            if coverage_count[idx] <= 1:
                safe = False
                break

        if safe:
            # Remove block
            for idx in block_tsets[min_idx]:
                coverage_count[idx] -= 1
            blocks.pop(min_idx)
            block_tsets.pop(min_idx)
            improved = True
        else:
            # Mark as non-removable by setting conductance high
            # Try next lowest
            # Simple approach: just break to avoid infinite loop
            # Actually try all blocks in order of conductance
            order = np.argsort(conductances)
            found = False
            for idx in order:
                if time.time() >= deadline:
                    break
                idx = int(idx)
                safe = True
                for t_idx in block_tsets[idx]:
                    if coverage_count[t_idx] <= 1:
                        safe = False
                        break
                if safe:
                    for t_idx in block_tsets[idx]:
                        coverage_count[t_idx] -= 1
                    blocks.pop(idx)
                    block_tsets.pop(idx)
                    improved = True
                    found = True
                    break
            if not found:
                break

    return blocks


def construct(v, k, t):
    """Main entry point: construct a covering design C(v,k,t)."""
    global _global_deadline
    start_time = time.time()
    _global_deadline = start_time + 145  # Safety margin before 150s limit

    # Precompute
    all_tsets, tset_to_idx = _precompute_tsets(v, t)
    n_tsets = len(all_tsets)

    rng = random.Random(42)
    best_blocks = None
    best_count = float('inf')

    # Phase 1: Multiple greedy restarts
    deadline_greedy = _global_deadline - 30
    num_restarts = 0
    while time.time() < deadline_greedy and num_restarts < 15:
        blocks = _greedy_cover_v2(v, k, t, all_tsets, tset_to_idx, rng, time.time() + 8)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
        num_restarts += 1

    if best_blocks is None:
        # Fallback: should not happen
        return []

    # Phase 2: Spectral pruning
    if time.time() < _global_deadline - 10:
        best_blocks = _spectral_prune(best_blocks, v, k, t, all_tsets, tset_to_idx, _global_deadline - 5)
        best_count = len(best_blocks)

    # Phase 3: Additional local search removals
    if time.time() < _global_deadline - 3:
        best_blocks = _local_search_remove(best_blocks, v, k, t, all_tsets, tset_to_idx, _global_deadline - 1)
        best_count = len(best_blocks)

    return best_blocks
