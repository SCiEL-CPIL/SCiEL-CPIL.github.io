import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    # Precomputation
    # 1. Generate all t-subsets and map them to indices
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    n_tsets = len(all_tsets)
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. For each block, find the indices of the t-subsets it covers
    # block_tsets[i] is an array of tset indices covered by block i
    block_tsets = np.zeros((n_blocks, k * (k-1) // 2), dtype=np.int32) # Max size, will trim
    # Actually, number of t-subsets in a k-block is C(k, t)
    num_t_per_block = 1
    for i in range(1, t+1):
        num_t_per_block *= (k - t + i) // i
        
    block_tsets = np.zeros((n_blocks, num_t_per_block), dtype=np.int32)
    
    for b_idx, block in enumerate(all_blocks):
        tsets_in_block = list(itertools.combinations(block, t))
        for t_idx, ts in enumerate(tsets_in_block):
            block_tsets[b_idx, t_idx] = tset_to_idx[ts]
            
    # State
    counts = np.zeros(n_tsets, dtype=np.int32)
    active_blocks = set()
    uncovered_mask = np.ones(n_tsets, dtype=bool) # All uncovered initially
    
    # Helper to update state
    def add_block(b_idx):
        active_blocks.add(b_idx)
        counts[block_tsets[b_idx]] += 1
        # Update uncovered mask only for affected tsets
        affected = block_tsets[b_idx]
        uncovered_mask[affected] = counts[affected] == 0
        return uncovered_mask.sum() > 0 # Return True if still uncovered exists
        
    def remove_block(b_idx):
        active_blocks.remove(b_idx)
        counts[block_tsets[b_idx]] -= 1
        affected = block_tsets[b_idx]
        uncovered_mask[affected] = counts[affected] == 0
        return uncovered_mask.sum() > 0

    def get_best_block():
        # Vectorized scoring: sum of uncovered_mask over tsets for each block
        # uncovered_mask[block_tsets] creates a (n_blocks, num_t_per_block) boolean array
        scores = np.sum(uncovered_mask[block_tsets], axis=1)
        best_idx = np.argmax(scores)
        if scores[best_idx] == 0:
            return -1
        return int(best_idx)

    def is_redundant(b_idx):
        # A block is redundant if all its tsets are covered by > 1 block
        return np.all(counts[block_tsets[b_idx]] > 1)

    # Phase 1: Initial Greedy Construction
    while np.any(uncovered_mask):
        best = get_best_block()
        if best == -1:
            break # Should not happen
        add_block(best)
        
    # Phase 2: Local Search
    start_time = time.time()
    deadline = start_time + 100.0 # 100s safety margin
    
    while time.time() < deadline:
        # Try to remove redundant blocks
        removed_any = False
        redundant_candidates = [b for b in list(active_blocks) if is_redundant(b)]
        
        if redundant_candidates:
            # Pick one randomly to allow exploration
            b_to_remove = random.choice(redundent_candidates)
            remove_block(b_to_remove)
            removed_any = True
        else:
            # Stuck in local optimum. Perturb by removing a random block and repairing.
            if active_blocks:
                b_to_remove = random.choice(list(active_blocks))
                remove_block(b_to_remove)
                removed_any = True
            else:
                break # No blocks, nothing to do (shouldn't happen for v=19)
                
        # Repair: Add blocks until valid
        while np.any(uncovered_mask):
            best = get_best_block()
            if best == -1:
                # Fallback: add any random block? Or break?
                # If no block covers uncovered sets, the problem is infeasible (not the case here)
                break
            add_block(best)
            
    # Convert active blocks (indices) back to lists of points
    result = [list(all_blocks[i]) for i in active_blocks]
    return result
