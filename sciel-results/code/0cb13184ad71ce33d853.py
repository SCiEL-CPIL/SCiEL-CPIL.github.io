import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 135  # Leave margin for 150s kill

    # Fallback for non-standard parameters
    if v != 19 or k != 8 or t != 5:
        return _fallback_greedy(v, k, t)

    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # 2. Precompute all k-blocks and the t-subsets they cover
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # block_covered_subsets[i] = list of indices of t-subsets covered by block i
    block_covered_subsets = []
    # blocks_containing_subset[j] = list of indices of k-blocks that contain t-subset j
    blocks_containing_subset = [[] for _ in range(n_t_subsets)]
    
    # Precompute the global index for each block tuple for fast lookup later
    block_to_global_idx = {block: i for i, block in enumerate(all_k_blocks)}
    
    for i, block in enumerate(all_k_blocks):
        covered_idx = []
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            covered_idx.append(idx)
            blocks_containing_subset[idx].append(i)
        block_covered_subsets.append(np.array(covered_idx, dtype=np.int32))
    
    def time_elapsed():
        return time.time() - start_time

    def greedy_cover(rng, strategy='random'):
        """
        Greedy cover.
        strategy: 'random' picks a random uncovered subset.
                  'min_candidates' picks the uncovered subset with the fewest candidate blocks.
                  'max_freq' picks the uncovered subset that appears in the most candidate blocks? No, usually min candidates is better for bottleneck.
        """
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_used = []
        
        # For 'min_candidates', we need to know the number of candidates for each uncovered subset.
        # This is static for the universe, but we only care about uncovered ones.
        # We can precompute the length of blocks_containing_subset for each index.
        # candidate_counts[j] = len(blocks_containing_subset[j])
        
        while np.any(uncovered):
            if time_elapsed() > time_limit - 5:
                break
            
            unc_idx = np.nonzero(uncovered)[0]
            if len(unc_idx) == 0:
                break
            
            if strategy == 'random':
                target_sub_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            elif strategy == 'min_candidates':
                # Find the uncovered subset with the fewest total possible blocks (static property)
                # This helps cover "hard" subsets early.
                # We can just pick the one with min len(blocks_containing_subset[idx])
                # To do this fast, we can precompute a priority or just scan.
                # Scanning 11628 items is fast enough?
                # Let's just pick random for now to keep it simple and fast, 
                # or maybe 'min_candidates' is better.
                # Let's try 'min_candidates' by finding the min of a precomputed array.
                # But we only want uncovered ones.
                # Let's stick to random for speed in the inner loop, 
                # but maybe alternate strategies in restarts.
                target_sub_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            else:
                target_sub_idx = unc_idx[rng.randint(0, len(unc_idx) - 1)]
            
            candidate_block_indices = blocks_containing_subset[target_sub_idx]
            
            if not candidate_block_indices:
                break
                
            best_block_idx = -1
            best_gain = -1
            
            # Optimization: Vectorized gain calculation?
            # We have a list of block indices. We need sum(uncovered[block_covered_subsets[b_idx]])
            # Since the number of candidates is small (~364), a loop is fine.
            for b_idx in candidate_block_indices:
                covered = block_covered_subsets[b_idx]
                gain = np.sum(uncovered[covered])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            blocks_used.append(all_k_blocks[best_block_idx])
            uncovered[block_covered_subsets[best_block_idx]] = False
            
        return blocks_used

    def try_remove_redundant(blocks):
        """
        Remove blocks that are redundant.
        """
        if not blocks:
            return blocks
            
        coverage_count = defaultdict(int)
        block_t_subsets_cache = []
        
        for block in blocks:
            b_tuple = tuple(block)
            g_idx = block_to_global_idx[b_tuple]
            ts_indices = block_covered_subsets[g_idx]
            block_t_subsets_cache.append(ts_indices)
            for idx in ts_indices:
                coverage_count[idx] += 1
        
        new_blocks = []
        for i, block in enumerate(blocks):
            ts_indices = block_t_subsets_cache[i]
            redundant = True
            for idx in ts_indices:
                if coverage_count[idx] <= 1:
                    redundant = False
                    break
            
            if not redundant:
                new_blocks.append(block)
            else:
                for idx in ts_indices:
                    coverage_count[idx] -= 1
        
        return new_blocks

    def local_search_swap(blocks, rng):
        """
        Try to swap a block for another that allows removing a different block?
        Or simply try to replace a block with another that covers the same unique parts?
        This is complex. Let's stick to redundancy removal for now.
        Maybe a simple hill-climbing: try replacing a block with a random other block
        that covers its unique subsets, and see if it allows further reduction?
        Too slow.
        """
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts with different strategies/seeds
    # Strategy 1: Random restarts
    seeds = [42, 123, 7, 99, 2024, 555, 111, 222, 333, 444, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for i, seed in enumerate(seeds):
        if time_elapsed() > time_limit - 10:
            break
            
        rng = random.Random(seed)
        
        # Alternate between random and 'min_candidates' logic if implemented
        # For now, just random.
        blocks = greedy_cover(rng)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
        # Polish
        if time_elapsed() < time_limit - 5:
            polished = try_remove_redundant(best_blocks)
            if len(polished) < best_count:
                best_count = len(polished)
                best_blocks = polished
                
                # Try again on the polished version?
                polished2 = try_remove_redundant(best_blocks)
                if len(polished2) < best_count:
                    best_count = len(polished2)
                    best_blocks = polished2

    # If we ran out of time or didn't find a full cover, ensure we return something valid
    if best_blocks is None:
        rng = random.Random(0)
        best_blocks = greedy_cover(rng)
        
    # Verification and Completion
    covered = np.zeros(n_t_subsets, dtype=bool)
    
    for block in best_blocks:
        g_idx = block_to_global_idx[tuple(block)]
        covered[block_covered_subsets[g_idx]] = True
        
    if not np.all(covered):
        # We have uncovered subsets. We must cover them.
        uncovered = ~covered
        while np.any(uncovered):
            if time_elapsed() > time_limit - 1:
                # Emergency: we have very little time.
                # Just pick the first uncovered and add a block.
                pass
                
            unc_idx = np.nonzero(uncovered)[0]
            if len(unc_idx) == 0: break
            target_sub_idx = unc_idx[0] # Pick first
            candidate_block_indices = blocks_containing_subset[target_sub_idx]
            
            best_block_idx = -1
            best_gain = -1
            for b_idx in candidate_block_indices:
                covered_arr = block_covered_subsets[b_idx]
                gain = np.sum(uncovered[covered_arr])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                break
                
            best_blocks.append(all_k_blocks[best_block_idx])
            uncovered[block_covered_subsets[best_block_idx]] = False

    return [list(b) for b in best_blocks]

def _fallback_greedy(v, k, t):
    """Simple fallback for non-standard parameters."""
    start_time = time.time()
    time_limit = 135
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    uncovered = set(all_t_subsets)
    blocks = []
    
    while uncovered:
        if time.time() - start_time > time_limit:
            break
        ts = min(uncovered)
        block = set(ts)
        while len(block) < k:
            best_point = -1
            best_gain = -1
            for x in range(v):
                if x in block:
                    continue
                gain = 0
                for rest in itertools.combinations(sorted(block), t - 1):
                    ts_new = tuple(sorted(rest + (x,)))
                    if ts_new in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.add(best_point)
        
        blocks.append(sorted(block))
        for covered_ts in itertools.combinations(block, t):
            uncovered.discard(covered_ts)
            
    return blocks
