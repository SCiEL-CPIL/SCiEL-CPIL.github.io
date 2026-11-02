import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precomputation
    # 1. Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Precompute t-subset indices for each block
    # block_t_indices[i] is a numpy array of shape (t_choose_k,) containing indices of t-subsets in block i
    t_choose_k = int(np.prod([k - t + i for i in range(t)]) / np.prod(range(1, t + 1))) # C(k,t)
    # Simpler: just compute C(k,t)
    from math import comb
    t_choose_k = comb(k, t)
    
    block_t_indices = np.zeros((num_blocks, t_choose_k), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        ts_list = list(itertools.combinations(block, t))
        for j, ts in enumerate(ts_list):
            block_t_indices[i, j] = t_subset_to_idx[ts]
            
    best_solution = None
    best_len = float('inf')
    
    # Run multiple restarts
    while time.time() < deadline:
        # Random seed for this restart
        # We can vary the tie-breaking strategy
        
        uncovered = np.ones(num_t, dtype=bool)
        coverage_counts = np.zeros(num_t, dtype=np.int32)
        solution_blocks = []
        used_block_indices = set()
        
        while True:
            if time.time() > deadline:
                break
            
            # Calculate gains
            # gains[i] = number of uncovered t-subsets covered by block i
            gains = uncovered[block_t_indices].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to explore different basins
            # If we always pick the first, we get the same solution.
            # Picking randomly helps escape local minima of the greedy choice.
            best_block_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            # Add block
            solution_blocks.append(best_block_idx)
            used_block_indices.add(best_block_idx)
            
            # Update uncovered and coverage counts
            # Mark t-subsets covered by this block
            covered_ts = block_t_indices[best_block_idx]
            newly_covered = covered_ts[uncovered[covered_ts]]
            
            # Update coverage counts for all t-subsets in this block
            coverage_counts[covered_ts] += 1
            
            # Update uncovered mask
            uncovered[covered_ts] = False
            
            # Check if all covered
            if not np.any(uncovered):
                break
        
        if time.time() > deadline:
            break
            
        # Check validity (should be covered by construction, but safety check)
        if np.any(uncovered):
            continue
            
        # Deletion Phase
        # Try to remove redundant blocks
        # We can iterate multiple times or just once in random order
        # Let's try removing blocks in reverse order of addition or random order
        # To be effective, we should check if a block is "essential".
        # A block is essential if it is the only block covering some t-subset.
        
        current_solution = solution_blocks[:]
        
        # Sort by some heuristic? Or just iterate.
        # Let's try iterating through the solution and removing if possible.
        # We need to update coverage counts if we remove a block.
        
        # Let's make a copy of coverage counts to allow rollback if we want, 
        # but simpler is to just check min count.
        
        # Actually, let's just do a single pass deletion.
        # Iterate through indices in current_solution
        # If we remove a block, we must decrement coverage_counts.
        
        # To be more aggressive, we can do multiple passes.
        # But time is limited. One pass is usually good.
        
        # Let's try a more robust deletion:
        # While we can remove a block:
        #   Find a block that has no unique t-subsets (i.e., min coverage in its t-subsets > 1)
        #   Remove it.
        
        # This might be slow if we search linearly each time.
        # Let's just do a single pass in reverse order.
        
        for i in range(len(current_solution) - 1, -1, -1):
            b_idx = current_solution[i]
            ts_indices = block_t_indices[b_idx]
            # Check if any t-subset in this block has coverage count == 1
            if np.min(coverage_counts[ts_indices]) > 1:
                # Safe to remove
                coverage_counts[ts_indices] -= 1
                current_solution.pop(i)
                
        # Update best solution
        if len(current_solution) < best_len:
            best_len = len(current_solution)
            best_solution = [list(all_blocks[b_idx]) for b_idx in current_solution]
            
        # If we found a very good solution, maybe stop early?
        # Theoretical lower bound is 131. If we hit 131, we can stop.
        if best_len <= 131:
            break

    if best_solution is None:
        # Fallback: return a valid but potentially larger solution
        # Just run one deterministic greedy to ensure validity
        uncovered = np.ones(num_t, dtype=bool)
        solution_blocks = []
        while np.any(uncovered):
            gains = uncovered[block_t_indices].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0: break
            best_block_idx = np.argmax(gains)
            solution_blocks.append(best_block_idx)
            covered_ts = block_t_indices[best_block_idx]
            uncovered[covered_ts] = False
        return [list(all_blocks[b_idx]) for b_idx in solution_blocks]
        
    return best_solution
