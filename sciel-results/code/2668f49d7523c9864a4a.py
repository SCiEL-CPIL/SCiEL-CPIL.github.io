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
            
    # 1. Greedy Initialization
    # Start with all t-subsets uncovered
    uncovered_indices = set(range(num_t_subsets))
    current_blocks = []
    
    # Randomized Greedy
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
                
    # Store initial greedy solution as fallback
    initial_greedy_blocks = list(current_blocks)
    
    # 2. Simulated Annealing
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
    
    while time.time() - start_time < timeout:
        if temp < min_temp:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Remove a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                bi = current_blocks[idx]
                
                # Calculate change in uncovered count
                # A t-subset becomes uncovered if coverage[ti] == 1
                add_uncovered = 0
                for ti in block_t_indices[bi]:
                    if coverage[ti] == 1:
                        add_uncovered += 1
                        
                new_uncovered = uncovered_count + add_uncovered
                new_cost = (len(current_blocks) - 1) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks.pop(idx)
                    for ti in block_t_indices[bi]:
                        coverage[ti] -= 1
                    uncovered_count = new_uncovered
                    current_cost = new_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = list(current_blocks)
                        
        elif move_type < 0.8:
            # Add a block
            bi = random.randint(0, num_blocks - 1)
            
            # Calculate change in uncovered count
            # A t-subset becomes covered if coverage[ti] == 0
            reduce_uncovered = 0
            for ti in block_t_indices[bi]:
                if coverage[ti] == 0:
                    reduce_uncovered += 1
                    
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
            # Swap a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                old_bi = current_blocks[idx]
                new_bi = random.randint(0, num_blocks - 1)
                
                if old_bi == new_bi:
                    continue
                    
                old_ts = block_t_indices[old_bi]
                new_ts = block_t_indices[new_bi]
                
                old_set = set(old_ts)
                new_set = set(new_ts)
                
                only_old = old_set - new_set
                only_new = new_set - old_set
                
                inc_uncovered = 0
                for ti in only_old:
                    if coverage[ti] == 1:
                        inc_uncovered += 1
                        
                dec_uncovered = 0
                for ti in only_new:
                    if coverage[ti] == 0:
                        dec_uncovered += 1
                        
                delta_uncovered = inc_uncovered - dec_uncovered
                new_uncovered = uncovered_count + delta_uncovered
                new_cost = len(current_blocks) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks[idx] = new_bi
                    for ti in only_old:
                        coverage[ti] -= 1
                    for ti in only_new:
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
        # Fallback to initial greedy
        best_blocks = initial_greedy_blocks

    result_blocks = [list(all_blocks[bi]) for bi in best_blocks]
    return result_blocks
