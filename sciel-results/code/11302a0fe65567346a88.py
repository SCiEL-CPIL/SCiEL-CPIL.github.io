"""Covering design C(v,k,t): minimize number of k-subset blocks covering all t-subsets.

Combined approach:
- Full enumeration of all k-blocks with vectorized coverage matrix (NumPy).
- Fast greedy set-cover using matrix-vector products.
- Vectorized redundancy removal (drop blocks that cover no critical t-subset).
- Local search: 1-for-1 and 2-for-1 swaps to escape local optima.
- Multiple restarts with different random seeds within the time budget.
"""
import time
import random
import itertools
import numpy as np

_CACHE = {}


def _prepare(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]

    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))

    n_blocks = len(all_blocks)
    n_tsubs = len(all_tsubs)

    # Map t-subset tuple to index
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Build coverage matrix: Cover[i, j] = 1 if tsub j is in block i
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    for i, B in enumerate(all_blocks):
        B_set = set(B)
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1

    _CACHE[key] = (all_blocks, Cover, n_blocks, n_tsubs)
    return _CACHE[key]


def _greedy_construct(Cover, n_tsubs, deadline, rng):
    """Greedy set cover. Returns (selected_indices, uncovered_mask)."""
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    selected = []
    selected_set = set()

    while True:
        if time.time() > deadline:
            break
        if not np.any(uncovered):
            break

        # Compute gains: how many currently uncovered t-subsets each block covers
        gains = Cover @ uncovered
        # Add small random perturbation for diversity
        gains_f = gains.astype(np.float64) + rng.random(n_blocks) * 1e-9
        best_idx = int(np.argmax(gains_f))

        if gains[best_idx] == 0:
            break

        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)

    return selected, uncovered


def _remove_redundant(selected, Cover, n_tsubs, deadline):
    """Iteratively remove blocks that cover no t-subset with count exactly 1."""
    if not selected:
        return selected

    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected:
        cover_counts += Cover[idx]

    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected = []
        for idx in selected:
            block_tsubs = np.where(Cover[idx] == 1)[0]
            if len(block_tsubs) > 0 and np.any(cover_counts[block_tsubs] == 1):
                new_selected.append(idx)
            else:
                cover_counts -= Cover[idx]
                changed = True
        selected = new_selected

    return selected


def _local_search(selected, Cover, n_tsubs, n_blocks, deadline, rng):
    """Local search with 1-for-1 swaps and 2-for-1 swaps."""
    if not selected:
        return selected

    selected_set = set(selected)
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected:
        cover_counts += Cover[idx]

    improved = True
    rounds = 0
    max_rounds = 500

    while improved and rounds < max_rounds and time.time() < deadline:
        improved = False
        rounds += 1

        # Try 1-for-1 swap: replace block i with block j if it doesn't increase count
        # and allows dropping i (i.e., i covers no critical t-subset after j is added)
        # Simpler: just try to find a block j not in selected that covers all critical
        # t-subsets of some selected block i, then swap.
        
        # First, try dropping redundant blocks
        new_selected = []
        for idx in selected:
            block_tsubs = np.where(Cover[idx] == 1)[0]
            if len(block_tsubs) == 0 or not np.any(cover_counts[block_tsubs] == 1):
                cover_counts -= Cover[idx]
                improved = True
            else:
                new_selected.append(idx)
        selected = new_selected
        selected_set = set(selected)

        if time.time() > deadline:
            break

        # Try 1-for-1 swaps
        swapped = False
        for pos in range(len(selected)):
            if time.time() > deadline:
                break
            i = selected[pos]
            # Find critical t-subsets of i (those with count == 1)
            crit_mask = Cover[i] * (cover_counts == 1)
            crit_tsubs = np.where(crit_mask > 0)[0]
            
            if len(crit_tsubs) == 0:
                continue

            # Find a block j not in selected that covers all critical t-subsets of i
            # and whose addition doesn't make any other t-subset problematic
            # For simplicity, just check if j covers all crit_tsubs
            for j in range(n_blocks):
                if j in selected_set:
                    continue
                if np.all(Cover[j, crit_tsubs] == 1):
                    # Check: after removing i and adding j, is coverage still valid?
                    # New counts: cover_counts - Cover[i] + Cover[j]
                    new_counts = cover_counts - Cover[i] + Cover[j]
                    if np.all(new_counts >= 1):
                        # Valid swap
                        cover_counts = new_counts
                        selected[pos] = j
                        selected_set.remove(i)
                        selected_set.add(j)
                        improved = True
                        swapped = True
                        break
            if swapped:
                break

        # Try 2-for-1 swaps: replace two blocks i1, i2 with one block j
        # This is more expensive, so do it less frequently
        if not swapped and rounds % 3 == 0 and time.time() < deadline:
            two_swapped = False
            for pos1 in range(len(selected)):
                if two_swapped or time.time() > deadline:
                    break
                for pos2 in range(pos1 + 1, len(selected)):
                    if two_swapped or time.time() > deadline:
                        break
                    i1 = selected[pos1]
                    i2 = selected[pos2]
                    # Critical t-subsets covered by i1 or i2
                    crit_mask = (Cover[i1] * (cover_counts == 1)) | (Cover[i2] * (cover_counts == 1))
                    crit_tsubs = np.where(crit_mask > 0)[0]
                    
                    if len(crit_tsubs) == 0:
                        continue

                    for j in range(n_blocks):
                        if j in selected_set:
                            continue
                        if np.all(Cover[j, crit_tsubs] == 1):
                            new_counts = cover_counts - Cover[i1] - Cover[i2] + Cover[j]
                            if np.all(new_counts >= 1):
                                cover_counts = new_counts
                                selected[pos1] = j
                                selected.pop(pos2)
                                selected_set.remove(i1)
                                selected_set.remove(i2)
                                selected_set.add(j)
                                improved = True
                                two_swapped = True
                                break
                    if two_swapped:
                        break
                if two_swapped:
                    break

    return selected


def construct(v, k, t):
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin before 150s kill

    all_blocks, Cover, n_blocks, n_tsubs = _prepare(v, k, t)

    best_selected = None
    best_count = float('inf')

    # Multiple restarts with different seeds
    seeds = [12345, 67890, 11111, 22222, 33333, 44444, 55555, 66666, 77777, 88888]
    rng_base = random.Random(42)

    # Reserve time for final local search on best solution
    restart_budget = deadline - 30.0  # Leave 30s for final refinement

    for seed in seeds:
        if time.time() > restart_budget:
            break
        rng = random.Random(seed)
        
        # Greedy construction
        selected, uncovered = _greedy_construct(Cover, n_tsubs, deadline - 5.0, rng)
        
        if np.any(uncovered):
            # Should not happen if greedy completed, but just in case
            continue

        # Remove redundant blocks
        selected = _remove_redundant(selected, Cover, n_tsubs, deadline - 10.0)
        
        # Local search
        selected = _local_search(selected, Cover, n_tsubs, n_blocks, deadline - 15.0, rng)
        
        count = len(selected)
        if count < best_count:
            best_count = count
            best_selected = list(selected)
        
        # Early termination if we hit a very good solution
        if best_count <= 140:
            break

    if best_selected is None:
        # Fallback: simple greedy
        rng = random.Random(12345)
        best_selected, _ = _greedy_construct(Cover, n_tsubs, deadline - 5.0, rng)
        best_selected = _remove_redundant(best_selected, Cover, n_tsubs, deadline - 2.0)

    # Final local search with remaining time
    best_selected = _local_search(best_selected, Cover, n_tsubs, n_blocks, deadline - 1.0, random.Random(99999))

    # Convert to list of blocks
    result = [list(all_blocks[i]) for i in best_selected]
    return result
