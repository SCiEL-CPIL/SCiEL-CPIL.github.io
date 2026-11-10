"""Covering design C(v,k,t) via fast vectorized greedy with randomized restarts and local refinement."""
import time
import random
import itertools
import numpy as np


def _build_lookup(v, t):
    """Precompute all t-subsets and a mapping from tuple to index."""
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    return all_tsubsets, tsubset_to_idx


def _fast_greedy(v, k, t, seed=42, max_time=100.0, randomize=False):
    """
    Fast greedy covering: repeatedly pick the k-subset covering the most uncovered t-subsets.
    Uses randomized tie-breaking when randomize=True.
    """
    rng = random.Random(seed)
    start_time = time.time()
    deadline = start_time + max_time
    
    all_tsubsets, tsubset_to_idx = _build_lookup(v, t)
    n_tsub = len(all_tsubsets)
    
    # covered[i] = True if t-subset i is covered
    covered = np.zeros(n_tsub, dtype=bool)
    num_uncovered = n_tsub
    
    blocks = []
    seen_blocks = set()
    
    while num_uncovered > 0:
        if time.time() > deadline:
            break
        
        # Find uncovered t-subset indices
        uncovered_indices = [i for i in range(n_tsub) if not covered[i]]
        if not uncovered_indices:
            break
        
        # Sample some uncovered t-subsets to build candidate blocks around
        if randomize and len(uncovered_indices) > 20:
            sample = rng.sample(uncovered_indices, min(20, len(uncovered_indices)))
        else:
            sample = uncovered_indices[:20]
        
        best_block = None
        best_gain = -1
        best_newly_covered = None
        
        for idx in sample:
            focus = all_tsubsets[idx]
            block_set = set(focus)
            
            # Greedily add points to maximize uncovered t-subsets covered
            while len(block_set) < k:
                best_point = None
                best_local_gain = -1
                second_best_point = None
                second_best_gain = -1
                
                for x in range(v):
                    if x in block_set:
                        continue
                    # Count uncovered t-subsets that would become covered by adding x
                    gain = 0
                    for rest in itertools.combinations(block_set, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        i = tsubset_to_idx.get(ts, -1)
                        if i != -1 and not covered[i]:
                            gain += 1
                    
                    if gain > best_local_gain:
                        second_best_point = best_point
                        second_best_gain = best_local_gain
                        best_local_gain = gain
                        best_point = x
                    elif gain == best_local_gain and gain > second_best_gain:
                        second_best_point = x
                        second_best_gain = gain
                    elif gain > second_best_gain:
                        second_best_point = x
                        second_best_gain = gain
                
                if best_point is None:
                    break
                
                # Randomized tie-breaking
                if randomize and second_best_point is not None and second_best_gain == best_local_gain and rng.random() < 0.3:
                    best_point = second_best_point
                
                block_set.add(best_point)
            
            block = tuple(sorted(block_set))
            
            if block in seen_blocks:
                continue
            
            # Count how many currently uncovered t-subsets this block covers
            newly_covered = []
            for ts in itertools.combinations(block, t):
                i = tsubset_to_idx.get(ts, -1)
                if i != -1 and not covered[i]:
                    newly_covered.append(i)
            
            gain = len(newly_covered)
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
                best_newly_covered = newly_covered
            elif gain == best_gain and best_block is not None and randomize and rng.random() < 0.2:
                best_block = block
                best_newly_covered = newly_covered
        
        if best_block is None:
            break
        
        seen_blocks.add(best_block)
        blocks.append(list(best_block))
        
        for i in best_newly_covered:
            if not covered[i]:
                covered[i] = True
                num_uncovered -= 1
    
    return blocks


def _local_refinement(blocks, v, k, t, max_time=10.0):
    """
    Try to improve the covering by removing redundant blocks.
    A block is redundant if all its t-subsets are covered by other blocks.
    """
    start_time = time.time()
    deadline = start_time + max_time
    
    if not blocks:
        return blocks
    
    all_tsubsets, tsubset_to_idx = _build_lookup(v, t)
    n_tsub = len(all_tsubsets)
    
    # For each t-subset, count how many blocks cover it
    tsub_count = np.zeros(n_tsub, dtype=np.int32)
    for block in blocks:
        for ts in itertools.combinations(block, t):
            i = tsubset_to_idx.get(ts, -1)
            if i != -1:
                tsub_count[i] += 1
    
    # Try to remove blocks that are redundant
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            block = blocks[i]
            # Check if all t-subsets of this block are covered by other blocks
            redundant = True
            for ts in itertools.combinations(block, t):
                idx = tsubset_to_idx.get(ts, -1)
                if idx != -1 and tsub_count[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                for ts in itertools.combinations(block, t):
                    idx = tsubset_to_idx.get(ts, -1)
                    if idx != -1:
                        tsub_count[idx] -= 1
                blocks.pop(i)
                improved = True
    
    return blocks


def _is_valid_covering(blocks, v, k, t):
    """Check if the blocks form a valid covering."""
    if not blocks:
        return v <= t  # trivial case
    
    all_tsubsets, tsubset_to_idx = _build_lookup(v, t)
    n_tsub = len(all_tsubsets)
    
    covered = np.zeros(n_tsub, dtype=bool)
    for block in blocks:
        if len(block) != k or len(set(block)) != k:
            return False
        for x in block:
            if x < 0 or x >= v:
                return False
        for ts in itertools.combinations(block, t):
            i = tsubset_to_idx.get(ts, -1)
            if i != -1:
                covered[i] = True
    
    return np.all(covered)


def construct(v, k, t):
    """
    Construct a covering design C(v,k,t).
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    total_start = time.time()
    total_deadline = total_start + 140.0  # Leave 10s safety margin before 150s kill
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple greedy strategies with different seeds
    seeds = [1, 42, 123, 777, 2024, 999, 555, 314, 271, 1618]
    
    for i, seed in enumerate(seeds):
        if time.time() > total_deadline:
            break
        
        remaining = total_deadline - time.time()
        if remaining < 3.0:
            break
        
        # Use more time for earlier runs, less for later ones
        if i < 3:
            alloc = min(remaining * 0.5, 25.0)
        elif i < 6:
            alloc = min(remaining * 0.3, 15.0)
        else:
            alloc = min(remaining * 0.2, 10.0)
        
        blocks = _fast_greedy(v, k, t, seed=seed, max_time=alloc, randomize=(i > 0))
        
        if not blocks:
            continue
        
        # Local refinement
        if time.time() < total_deadline - 5.0:
            blocks = _local_refinement(blocks, v, k, t, max_time=min(5.0, total_deadline - time.time() - 2.0))
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # Check if we've reached a good solution
        if best_len <= 300:
            break
    
    # Final safety check
    if best_blocks is None:
        # Fallback: simple greedy without time pressure
        best_blocks = _fast_greedy(v, k, t, seed=42, max_time=5.0, randomize=False)
    
    # Verify validity
    if not _is_valid_covering(best_blocks, v, k, t):
        # If invalid, try one more clean run
        if time.time() < total_deadline - 10.0:
            best_blocks = _fast_greedy(v, k, t, seed=42, max_time=min(10.0, total_deadline - time.time() - 5.0), randomize=False)
            if not _is_valid_covering(best_blocks, v, k, t):
                # Last resort: very conservative approach
                best_blocks = _fast_greedy(v, k, t, seed=1, max_time=5.0, randomize=False)
    
    return best_blocks
