"""Simulated annealing approach for covering design C(19,7,4) with efficient state updates."""
import itertools
import random
import math
import time
import sys
import numpy as np

# Use a fixed seed for reproducibility within a single run, but allow randomness
random.seed(42)
np.random.seed(42)

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Leave 10 seconds buffer
    
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and the t-subsets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # block_t_indices[i] is a list of t-subset indices covered by block i
    block_t_indices = []
    for b in all_blocks:
        t_idxs = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_t_indices.append(t_idxs)
        
    # Precompute t_subset_to_blocks for greedy initialization
    t_subset_to_blocks = [[] for _ in range(num_t_subsets)]
    for bi, t_idxs in enumerate(block_t_indices):
        for ti in t_idxs:
            t_subset_to_blocks[ti].append(bi)
            
    def greedy_init():
        """Randomized Greedy Initialization"""
        uncovered_indices = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered_indices:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered_indices))
            # Pick the best block from those covering this target
            candidate_blocks = t_subset_to_blocks[target_idx]
            best_block_idx = -1
            best_count = 0
            
            for bi in candidate_blocks:
                # Count how many uncovered t-subsets this block covers
                count = 0
                for ti in block_t_indices[bi]:
                    if ti in uncovered_indices:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block_idx = bi
                    
            if best_block_idx == -1:
                break # Should not happen
                
            current_blocks.append(best_block_idx)
            
            # Update uncovered set
            for ti in block_t_indices[best_block_idx]:
                if ti in uncovered_indices:
                    uncovered_indices.remove(ti)
                    
        return current_blocks

    # Run multiple greedy initializations to find a good starting point
    best_initial_blocks = []
    best_initial_len = float('inf')
    for _ in range(10):
        init_blocks = greedy_init()
        if len(init_blocks) < best_initial_len:
            best_initial_len = len(init_blocks)
            best_initial_blocks = init_blocks
            
    current_blocks = best_initial_blocks
    
    # State: current_blocks (list of block indices)
    # Coverage: array of counts
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    for bi in current_blocks:
        for ti in block_t_indices[bi]:
            coverage[ti] += 1
            
    uncovered_count = np.sum(coverage == 0)
    current_cost = len(current_blocks) + 1000 * uncovered_count
    
    best_blocks = list(current_blocks)
    best_cost = current_cost
    
    temp = 50.0
    cooling = 0.995
    min_temp = 1.0
    
    # Precompute unique coverage helper? 
    # For "Swap" move: We want to replace block A with block B.
    # Condition: All t-subsets uniquely covered by A must be covered by B.
    # i.e., for all ts in A, if coverage[ts] == 1, then ts must be in B.
    
    while time.time() - start_time < timeout:
        if temp < min_temp:
            break
            
        move_type = random.random()
        
        if move_type < 0.4:
            # Remove a block (if safe)
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                bi = current_blocks[idx]
                
                # Check if safe to remove: all t-subsets in bi have coverage > 1
                safe = True
                for ti in block_t_indices[bi]:
                    if coverage[ti] == 1:
                        safe = False
                        break
                
                if safe:
                    # Calculate change
                    # Removing a safe block reduces cost by 1, uncovered count unchanged
                    new_cost = current_cost - 1
                    delta = new_cost - current_cost
                    
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        # Apply move
                        current_blocks.pop(idx)
                        for ti in block_t_indices[bi]:
                            coverage[ti] -= 1
                        current_cost = new_cost
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                            
        elif move_type < 0.7:
            # Add a block
            # Prefer blocks that cover uncovered t-subsets
            if uncovered_count > 0:
                # Pick a random uncovered t-subset
                # To do this efficiently, we can pick a random index and check, or maintain a list
                # For simplicity in this loop, we pick a random block and see if it helps
                bi = random.randint(0, num_blocks - 1)
                
                # Calculate change
                reduce_uncovered = 0
                for ti in block_t_indices[bi]:
                    if coverage[ti] == 0:
                        reduce_uncovered += 1
                        
                if reduce_uncovered > 0:
                    new_uncovered = uncovered_count - reduce_uncovered
                    new_cost = (len(current_blocks) + 1) + 1000 * new_uncovered
                    
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        # Apply move
                        current_blocks.append(bi)
                        for ti in block_t_indices[bi]:
                            coverage[ti] += 1
                        uncovered_count = new_uncovered
                        current_cost = new_cost
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
            else:
                # If no uncovered, just add random to explore? 
                # Or skip. Adding random when valid only increases cost, so likely rejected unless temp is high.
                bi = random.randint(0, num_blocks - 1)
                new_cost = current_cost + 1
                delta = 1
                if random.random() < math.exp(-delta / temp):
                    current_blocks.append(bi)
                    for ti in block_t_indices[bi]:
                        coverage[ti] += 1
                    current_cost = new_cost
        else:
            # Swap a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                old_bi = current_blocks[idx]
                
                # Find t-subsets uniquely covered by old_bi
                unique_ts = []
                for ti in block_t_indices[old_bi]:
                    if coverage[ti] == 1:
                        unique_ts.append(ti)
                
                # We want a new_bi that covers all unique_ts
                # If unique_ts is empty, any new_bi is a valid swap (maintains validity)
                # If unique_ts is not empty, new_bi must contain all of them.
                
                # Efficient way: 
                # If unique_ts is large, few blocks cover them.
                # We can pick a random candidate new_bi and check.
                
                # Optimization: If unique_ts is not empty, we can pick the first unique_ts 
                # and iterate over blocks covering it.
                
                candidate_new_bi = -1
                if not unique_ts:
                    candidate_new_bi = random.randint(0, num_blocks - 1)
                    if candidate_new_bi == old_bi:
                        candidate_new_bi = (candidate_new_bi + 1) % num_blocks
                else:
                    # Pick a random unique ts and find a block covering it
                    target_ts = random.choice(unique_ts)
                    candidates = t_subset_to_blocks[target_ts]
                    if candidates:
                        # Pick a random candidate from those
                        candidate_new_bi = random.choice(candidates)
                        # Check if it covers ALL unique_ts
                        new_ts_set = set(block_t_indices[candidate_new_bi])
                        valid = all(ts in new_ts_set for ts in unique_ts)
                        if not valid or candidate_new_bi == old_bi:
                            # Try a few more? Or just skip this move
                            candidate_new_bi = -1
                            # If we fail to find a valid swap, we can try to "break" the swap
                            # by allowing it to uncover some? 
                            # The directive says "Swap for one that covers the same unique t-subsets".
                            # So if we can't find one, we skip.
                            pass
                
                if candidate_new_bi != -1 and candidate_new_bi != old_bi:
                    new_bi = candidate_new_bi
                    new_ts = block_t_indices[new_bi]
                    
                    # Calculate delta
                    # Remove old_bi, Add new_bi
                    # Change in uncovered:
                    # 1. ts in old_bi but not in new_bi: if coverage[ts] == 1, it becomes uncovered (+1)
                    # 2. ts in new_bi but not in old_bi: if coverage[ts] == 0, it becomes covered (-1)
                    
                    old_set = set(block_t_indices[old_bi])
                    new_set = set(new_ts)
                    
                    inc_uncovered = 0
                    for ti in (old_set - new_set):
                        if coverage[ti] == 1:
                            inc_uncovered += 1
                            
                    dec_uncovered = 0
                    for ti in (new_set - old_set):
                        if coverage[ti] == 0:
                            dec_uncovered += 1
                            
                    delta_uncovered = inc_uncovered - dec_uncovered
                    new_uncovered = uncovered_count + delta_uncovered
                    new_cost = len(current_blocks) + 1000 * new_uncovered
                    
                    delta = new_cost - current_cost
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        # Apply move
                        current_blocks[idx] = new_bi
                        for ti in (old_set - new_set):
                            coverage[ti] -= 1
                        for ti in (new_set - old_set):
                            coverage[ti] += 1
                            
                        uncovered_count = new_uncovered
                        current_cost = new_cost
                        
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_blocks = list(current_blocks)
                        
        temp *= cooling
        
    # Verify and return
    # Ensure best_blocks is valid
    test_cov = np.zeros(num_t_subsets, dtype=np.int32)
    for bi in best_blocks:
        for ti in block_t_indices[bi]:
            test_cov[ti] += 1
            
    if np.any(test_cov == 0):
        # Fallback to best initial greedy
        best_blocks = best_initial_blocks

    result_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    return result_blocks
