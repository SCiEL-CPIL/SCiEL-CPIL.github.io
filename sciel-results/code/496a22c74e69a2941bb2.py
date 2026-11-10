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

# Global cache for precomputed data
_PRECOMPUTED = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
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
            
    # Precompute block_t_indices as numpy arrays for faster operations
    block_t_indices_np = [np.array(t_idxs, dtype=np.int32) for t_idxs in block_t_indices]
    
    _PRECOMPUTED[key] = (all_t_subsets, num_t_subsets, t_subset_to_idx, 
                         all_blocks, num_blocks, block_t_indices, 
                         t_subset_to_blocks, block_t_indices_np)
    return _PRECOMPUTED[key]

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Leave 10 seconds buffer
    
    # Use precomputed data
    (all_t_subsets, num_t_subsets, t_subset_to_idx, 
     all_blocks, num_blocks, block_t_indices, 
     t_subset_to_blocks, block_t_indices_np) = _precompute(v, k, t)
    
    # 1. Greedy Initialization with tie-breaking
    # Start with all t-subsets uncovered
    uncovered_indices = set(range(num_t_subsets))
    current_blocks = []
    
    # Randomized Greedy with better tie-breaking
    while uncovered_indices:
        # Pick a random uncovered t-subset
        target_idx = random.choice(list(uncovered_indices))
        # Pick the best block from those covering this target
        candidate_blocks = t_subset_to_blocks[target_idx]
        best_block_idx = -1
        best_count = 0
        best_second_count = 0
        
        for bi in candidate_blocks:
            # Count how many uncovered t-subsets this block covers
            count = 0
            for ti in block_t_indices[bi]:
                if ti in uncovered_indices:
                    count += 1
            if count > best_count or (count == best_count and random.random() < 0.5):
                best_second_count = best_count
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
            
    uncovered_count = int(np.sum(coverage == 0))
    current_cost = len(current_blocks) + 1000 * uncovered_count
    
    best_blocks = list(current_blocks)
    best_cost = current_cost
    
    # Local search: remove redundant blocks efficiently
    def local_search_remove_redundant(blocks, cov):
        """Remove blocks that don't have any unique t-subsets (coverage==1)."""
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks) - 1, -1, -1):
                bi = blocks[i]
                t_idxs = block_t_indices_np[bi]
                # Check if any t-subset has coverage == 1
                if np.any(cov[t_idxs] == 1):
                    continue
                # This block is redundant, remove it
                cov[t_idxs] -= 1
                blocks.pop(i)
                changed = True
        return blocks
    
    # Initial local search
    current_blocks = local_search_remove_redundant(current_blocks, coverage)
    uncovered_count = int(np.sum(coverage == 0))
    current_cost = len(current_blocks) + 1000 * uncovered_count
    
    if current_cost < best_cost:
        best_cost = current_cost
        best_blocks = list(current_blocks)
    
    temp = 50.0
    cooling = 0.995
    min_temp = 0.05
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < timeout:
        if temp < min_temp:
            break
            
        # Weighted move selection based on current state
        if uncovered_count == 0:
            # Focus on removal when fully covered
            move_type = random.random()
            if move_type < 0.6:
                move_choice = 'remove'
            elif move_type < 0.8:
                move_choice = 'swap'
            else:
                move_choice = 'add'
        else:
            # Mix moves when not fully covered
            move_type = random.random()
            if move_type < 0.3:
                move_choice = 'remove'
            elif move_type < 0.6:
                move_choice = 'add'
            else:
                move_choice = 'swap'
        
        accepted = False
        
        if move_choice == 'remove':
            # Remove a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                bi = current_blocks[idx]
                t_idxs = block_t_indices_np[bi]
                
                # Calculate change in uncovered count
                add_uncovered = int(np.sum(coverage[t_idxs] == 1))
                
                new_uncovered = uncovered_count + add_uncovered
                new_cost = (len(current_blocks) - 1) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks.pop(idx)
                    coverage[t_idxs] -= 1
                    uncovered_count = new_uncovered
                    current_cost = new_cost
                    accepted = True
                    
        elif move_choice == 'add':
            # Add a block - prefer blocks that cover many uncovered subsets
            # Sample a few candidates and pick the best
            num_samples = min(5, num_blocks)
            best_bi = -1
            best_reduce = -1
            for _ in range(num_samples):
                bi = random.randint(0, num_blocks - 1)
                t_idxs = block_t_indices_np[bi]
                reduce_uncovered = int(np.sum(coverage[t_idxs] == 0))
                if reduce_uncovered > best_reduce:
                    best_reduce = reduce_uncovered
                    best_bi = bi
            
            if best_bi >= 0:
                bi = best_bi
                t_idxs = block_t_indices_np[bi]
                
                # Calculate change in uncovered count
                reduce_uncovered = int(np.sum(coverage[t_idxs] == 0))
                
                new_uncovered = uncovered_count - reduce_uncovered
                new_cost = (len(current_blocks) + 1) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks.append(bi)
                    coverage[t_idxs] += 1
                    uncovered_count = new_uncovered
                    current_cost = new_cost
                    accepted = True
                    
        else:
            # Swap a block
            if len(current_blocks) > 0:
                idx = random.randint(0, len(current_blocks) - 1)
                old_bi = current_blocks[idx]
                new_bi = random.randint(0, num_blocks - 1)
                
                if old_bi == new_bi:
                    continue
                    
                old_ts = block_t_indices_np[old_bi]
                new_ts = block_t_indices_np[new_bi]
                
                # Efficient swap evaluation without set operations
                # Count t-subsets in old_ts with cov==1 (will become uncovered)
                inc_uncovered = int(np.sum(coverage[old_ts] == 1))
                # Count t-subsets in new_ts with cov==0 (will become covered)
                dec_uncovered = int(np.sum(coverage[new_ts] == 0))
                
                # Adjust for overlap: t-subsets in both old_ts and new_ts
                # For those, removing old and adding new doesn't change coverage
                # So we need to subtract overlaps from both counts
                # Find intersection
                old_set = set(old_ts)
                new_set = set(new_ts)
                overlap = old_set & new_set
                
                if overlap:
                    overlap_list = list(overlap)
                    # For overlapping t-subsets, coverage doesn't change
                    # So adjust inc and dec
                    inc_overlap = int(np.sum(coverage[overlap_list] == 1))
                    dec_overlap = int(np.sum(coverage[overlap_list] == 0))
                    inc_uncovered -= inc_overlap
                    dec_uncovered -= dec_overlap
                
                delta_uncovered = inc_uncovered - dec_uncovered
                new_uncovered = uncovered_count + delta_uncovered
                new_cost = len(current_blocks) + 1000 * new_uncovered
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    # Apply move
                    current_blocks[idx] = new_bi
                    coverage[old_ts] -= 1
                    coverage[new_ts] += 1
                    
                    uncovered_count = new_uncovered
                    current_cost = new_cost
                    accepted = True
        
        if accepted:
            # Local search after any accepted move
            current_blocks = local_search_remove_redundant(current_blocks, coverage)
            uncovered_count = int(np.sum(coverage == 0))
            current_cost = len(current_blocks) + 1000 * uncovered_count
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_blocks = list(current_blocks)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Restart temperature if stuck
            if no_improve_count > max_no_improve:
                temp = 30.0
                no_improve_count = 0
        else:
            no_improve_count += 1
        
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
