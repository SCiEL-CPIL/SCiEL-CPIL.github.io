import itertools
import numpy as np
import time
import random

def _precompute(v, k, t):
    """Precompute t-subset indices and block-to-t-subset mapping."""
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Local t-combinations within a k-set
    local_t_combos = list(itertools.combinations(range(k), t))
    n_t_in_k = len(local_t_combos)
    
    # Build block_t_indices: (n_k, n_t_in_k)
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    for j, lc in enumerate(local_t_combos):
        lc_list = list(lc)
        # For each k-subset, extract the t points at positions lc_list
        # and look up in t_sub_to_idx
        for i, block in enumerate(k_subsets):
            t_combo = tuple(block[p] for p in lc_list)
            block_t_indices[i, j] = t_sub_to_idx[t_combo]
    
    # Build inverted index: for each t-subset, list of blocks containing it
    # This allows efficient gain updates
    t_to_blocks = [[] for _ in range(n_t)]
    for i in range(n_k):
        for j in range(n_t_in_k):
            t_idx = block_t_indices[i, j]
            t_to_blocks[t_idx].append(i)
    
    return t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks


def _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, seed=0):
    """Greedy set cover with incremental gain updates using inverted index."""
    rng = random.Random(seed)
    
    uncovered = np.ones(n_t, dtype=bool)
    selected_blocks = []
    
    # gains[i] = number of uncovered t-subsets in block i
    gains = np.full(n_k, n_t_in_k, dtype=np.int32)
    
    # Precompute t_to_blocks as a list of arrays for faster indexing
    t_to_blocks_arr = [np.array(b, dtype=np.int32) if b else np.array([], dtype=np.int32) for b in t_to_blocks]
    
    while True:
        if time.time() - start_time > deadline - 10.0:
            break
            
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with max gain, pick randomly among them
        # Occasionally pick from near-max to diversify
        if rng.random() < 0.1 and max_gain > 1:
            # Pick from blocks with gain >= max_gain - 1
            threshold = max_gain - 1
            candidates = np.where(gains >= threshold)[0]
        else:
            candidates = np.where(gains == max_gain)[0]
        best_idx = candidates[rng.randint(0, len(candidates) - 1)]
        
        selected_blocks.append(best_idx)
        
        # Update uncovered and gains incrementally
        block_ts = block_t_indices[best_idx]
        newly_covered = []
        for t_idx in block_ts:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                newly_covered.append(t_idx)
        
        # Batch decrement gains for all blocks containing newly covered t-subsets
        if newly_covered:
            # Collect all block indices to decrement
            all_blocks_to_decr = []
            for t_idx in newly_covered:
                arr = t_to_blocks_arr[t_idx]
                if len(arr) > 0:
                    all_blocks_to_decr.append(arr)
            if all_blocks_to_decr:
                combined = np.concatenate(all_blocks_to_decr)
                np.add.at(gains, combined, -1)
        
    return selected_blocks


def _remove_redundant(selected_block_indices, block_t_indices, n_t, start_time, deadline):
    """Remove redundant blocks using precomputed indices."""
    if not selected_block_indices:
        return selected_block_indices
    
    # Build coverage count
    coverage_count = np.zeros(n_t, dtype=np.int32)
    for bi in selected_block_indices:
        coverage_count[block_t_indices[bi]] += 1
    
    remaining = list(selected_block_indices)
    changed = True
    while changed:
        if time.time() - start_time > deadline - 3.0:
            break
        changed = False
        i = 0
        while i < len(remaining):
            if time.time() - start_time > deadline - 3.0:
                return remaining
            bi = remaining[i]
            ts = block_t_indices[bi]
            if np.all(coverage_count[ts] > 1):
                coverage_count[ts] -= 1
                remaining.pop(i)
                changed = True
                # Don't increment i, check the new element at this position
            else:
                i += 1
    
    return remaining


def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    
    # Precompute
    t_subsets, t_sub_to_idx, n_t, k_subsets, n_k, block_t_indices, n_t_in_k, t_to_blocks = _precompute(v, k, t)
    
    # Multiple restarts with different seeds
    best_result = None
    best_count = float('inf')
    
    seed = 0
    max_restarts = 50
    while seed < max_restarts:
        if time.time() - start_time > deadline - 15.0:
            break
        
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, seed)
        
        # Remove redundant blocks
        selected = _remove_redundant(selected, block_t_indices, n_t, start_time, deadline)
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_result = list(selected)
        
        # Check time for next restart
        if time.time() - start_time > deadline - 20.0:
            break
        
        seed += 1
    
    # Use remaining time for additional local search on the best result
    if best_result is not None and time.time() - start_time < deadline - 5.0:
        # Try to swap out blocks for better ones
        coverage_count = np.zeros(n_t, dtype=np.int32)
        for bi in best_result:
            coverage_count[block_t_indices[bi]] += 1
        
        improved = True
        while improved and time.time() - start_time < deadline - 5.0:
            improved = False
            for i in range(len(best_result)):
                if time.time() - start_time > deadline - 5.0:
                    break
                bi = best_result[i]
                ts = block_t_indices[bi]
                # Check if this block is redundant
                if np.all(coverage_count[ts] > 1):
                    coverage_count[ts] -= 1
                    best_result.pop(i)
                    improved = True
                    break
    
    if best_result is None:
        # Fallback
        selected = _greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, t_to_blocks, start_time, deadline, 0)
        selected = _remove_redundant(selected, block_t_indices, n_t, start_time, deadline)
        best_result = selected
    
    return [list(k_subsets[bi]) for bi in best_result]
