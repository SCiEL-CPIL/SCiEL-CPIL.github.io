"""Covering design C(v,k,t): vectorized greedy + redundancy removal via NumPy."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def _subset_to_idx(t_subs):
    return {s: i for i, s in enumerate(t_subs)}


def _precompute_gain_matrix(v, k, t, idx_map, n_tsubs):
    """Precompute gain_matrix[x, t1_idx] = t-subset index or -1.
    
    t1_all: list of all (t-1)-subsets
    gain_matrix: 2D numpy array of shape (v, len(t1_all)), dtype=int32
    gain_matrix[x, j] = idx_map[t1_all[j] + (x,)] if x not in t1_all[j], else -1
    """
    t1_all = list(itertools.combinations(range(v), t - 1))
    n_t1 = len(t1_all)
    gain_matrix = np.full((v, n_t1), -1, dtype=np.int32)
    for x in range(v):
        for j, s in enumerate(t1_all):
            if x not in s:
                new_sub = tuple(sorted(s + (x,)))
                if new_sub in idx_map:
                    gain_matrix[x, j] = idx_map[new_sub]
    return gain_matrix, t1_all


def _greedy_pass(v, k, t, rng, gain_matrix, t1_all, idx_map, n_tsubs, t_subs, deadline=None):
    """Vectorized greedy pass using precomputed gain matrix."""
    uncovered = np.ones(n_tsubs, dtype=bool)
    uncovered_count = n_tsubs
    blocks = []

    # Precompute t1_all as numpy array for fast lookup
    t1_arr = np.array(t1_all, dtype=np.int32)  # shape (n_t1, t-1)
    n_t1 = len(t1_all)

    while uncovered_count > 0:
        if deadline is not None and time.time() > deadline:
            break
        # Pick random uncovered tsub
        while True:
            idx = rng.randrange(n_tsubs)
            if uncovered[idx]:
                break
        start_sub = t_subs[idx]
        block = list(start_sub)
        block_set = set(block)

        while len(block) < k:
            # Get all (t-1)-subsets of current block
            t1_subs_block = list(itertools.combinations(block, t - 1))
            
            # For each candidate x, compute gain using vectorized lookup
            candidates = [x for x in range(v) if x not in block_set]
            best_x = None
            best_gain = -1

            # Build lookup: for each t1-subset in block, find its index in t1_all
            # We need to find which t1_all entries match each t1_sub in t1_subs_block
            # Since t1_all is all combinations, we can use idx_map approach
            # Actually, let's just iterate candidates and compute gain
            for x in candidates:
                gain = 0
                row = gain_matrix[x]  # shape (n_t1,)
                for rest in t1_subs_block:
                    # Find index of rest in t1_all
                    # This is expensive; let's use a different approach
                    pass
                
                # Alternative: precompute a mapping from t1-subset to index
                # Let's build it once outside
                pass

            # Simpler approach: for each candidate, count uncovered tsubs
            # that would be covered by adding x to block
            for x in candidates:
                gain = 0
                # t-subsets covered by block ∪ {x} that involve x
                # = {s ∪ {x} : s is (t-1)-subset of block}
                for rest in t1_subs_block:
                    new_sub = tuple(sorted(rest + (x,)))
                    i = idx_map.get(new_sub, -1)
                    if i >= 0 and uncovered[i]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x

            block.append(best_x)
            block_set.add(best_x)

        block.sort()
        blocks.append(block)

        # Update uncovered
        for comb in itertools.combinations(block, t):
            i = idx_map[comb]
            if uncovered[i]:
                uncovered[i] = False
                uncovered_count -= 1

    return blocks


def _remove_redundant_blocks(blocks, v, k, t, idx_map, n_tsubs, rng=None):
    """Remove redundant blocks using vectorized coverage counting."""
    if not blocks:
        return blocks
    if rng is None:
        rng = random.Random(0)

    n_blocks = len(blocks)
    
    # Build coverage matrix: coverage[j] = number of blocks covering tsub j
    coverage = np.zeros(n_tsubs, dtype=np.int32)
    block_tsub_lists = []
    for b in blocks:
        tsub_indices = []
        for comb in itertools.combinations(b, t):
            tsub_indices.append(idx_map[comb])
        block_tsub_lists.append(tsub_indices)
        for i in tsub_indices:
            coverage[i] += 1

    kept = list(range(n_blocks))
    changed = True
    max_passes = 10

    for _ in range(max_passes):
        if not changed:
            break
        changed = False
        rng.shuffle(kept)

        new_kept = []
        for i in kept:
            redundant = True
            for tsub_idx in block_tsub_lists[i]:
                if coverage[tsub_idx] <= 1:
                    redundant = False
                    break
            if not redundant:
                new_kept.append(i)
            else:
                for tsub_idx in block_tsub_lists[i]:
                    coverage[tsub_idx] -= 1
                changed = True

        kept = new_kept

    return [blocks[i] for i in kept]


def _swap_local_search(blocks, v, k, t, idx_map, n_tsubs, rng, max_iters=50):
    """Redundancy removal with randomization (same as _remove_redundant_blocks)."""
    return _remove_redundant_blocks(blocks, v, k, t, idx_map, n_tsubs, rng)


def construct(v, k, t):
    """Find a covering design using vectorized greedy + redundancy removal."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin

    t_subs = _all_t_subsets(v, t)
    idx_map = _subset_to_idx(t_subs)
    n_tsubs = len(t_subs)

    # Precompute gain matrix
    gain_matrix, t1_all = _precompute_gain_matrix(v, k, t, idx_map, n_tsubs)

    best_blocks = None
    best_len = float('inf')

    attempt = 0
    while time.time() < deadline:
        rng = random.Random(1000 + attempt)
        blocks = _greedy_pass(v, k, t, rng, gain_matrix, t1_all, idx_map, n_tsubs, t_subs, deadline)
        
        # Check if complete
        if blocks:
            covered = np.zeros(n_tsubs, dtype=bool)
            for b in blocks:
                for comb in itertools.combinations(b, t):
                    covered[idx_map[comb]] = True
            if not covered.all():
                # Incomplete due to deadline; skip
                attempt += 1
                continue
        
        # Redundancy removal
        blocks = _remove_redundant_blocks(blocks, v, k, t, idx_map, n_tsubs, rng)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]

        if best_len <= 338:
            break

        attempt += 1

    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = _greedy_pass(v, k, t, rng, gain_matrix, t1_all, idx_map, n_tsubs, t_subs)
        best_blocks = _remove_redundant_blocks(best_blocks, v, k, t, idx_map, n_tsubs, rng)

    return best_blocks
# EVOLVE-BLOCK-END
