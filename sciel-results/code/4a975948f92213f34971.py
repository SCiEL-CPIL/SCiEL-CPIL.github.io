import itertools
import random
import time
import numpy as np
from math import comb

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    n_tsubsets = comb(v, t)
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute for each point: bitmask of t-subsets containing it
    # Using numpy bitarrays for speed
    point_masks = [np.zeros(n_tsubsets, dtype=bool) for _ in range(v)]
    for i, ts in enumerate(all_tsubsets):
        for p in ts:
            point_masks[p][i] = True

    # Precompute for each point pair, triple, etc. to speed up gain calculation if needed, 
    # but for Simulated Annealing we mainly need fast coverage updates.
    
    # For a block B, the t-subsets it covers:
    # We need a fast way to get the list of t-subset indices for a given block.
    # Since v=19, k=7, t=4, C(7,4)=35.
    # We can precompute the list of t-subset indices for every possible block?
    # C(19,7) = 50388 blocks. 50388 * 35 = 1.76M entries. This is feasible.
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    block_to_idx = {block: i for i, block in enumerate(all_blocks)}
    
    # block_cover_indices[block_idx] = list of t-subset indices covered by this block
    block_cover_indices = []
    for block in all_blocks:
        indices = [tsubset_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_cover_indices.append(indices)
        
    # 1. Generate an initial feasible solution using a fast greedy method
    # We will use a weighted greedy that picks the block covering the most uncovered t-subsets.
    # To make it fast, we maintain an array of uncovered counts.
    
    def generate_initial_solution():
        uncovered = np.ones(n_tsubsets, dtype=np.int32)
        selected_blocks = []
        
        # To speed up greedy, we can precompute for each t-subset which blocks cover it.
        # t_subset_to_blocks[idx] = list of block indices
        t_subset_to_blocks = [[] for _ in range(n_tsubsets)]
        for b_idx, block in enumerate(all_blocks):
            for ts_idx in block_cover_indices[b_idx]:
                t_subset_to_blocks[ts_idx].append(b_idx)
                
        while np.any(uncovered > 0):
            if time.time() > deadline - 20:
                break
                
            # Find an uncovered t-subset with the fewest covering blocks (most constrained)
            # This is a heuristic to avoid wasting blocks on "easy" t-subsets
            # But finding the min count over all uncovered is O(n_tsubsets * avg_degree)
            # n_tsubsets = 3876. avg_degree ~ 1000. 3.8M ops. Feasible per step?
            # Number of steps ~ 150. 150 * 3.8M = 570M ops. Might be too slow in Python.
            
            # Alternative: Pick a random uncovered t-subset.
            uncovered_indices = np.where(uncovered > 0)[0]
            if len(uncovered_indices) == 0:
                break
            target_idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
            
            # Pick a block covering this t-subset.
            # Heuristic: Pick the block that covers the most OTHER uncovered t-subsets.
            candidates = t_subset_to_blocks[target_idx]
            if not candidates:
                # Should not happen if we cover all
                break
                
            best_block = -1
            best_gain = -1
            for b_idx in candidates:
                gain = 0
                for ts_idx in block_cover_indices[b_idx]:
                    if uncovered[ts_idx] > 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            selected_blocks.append(best_block)
            
            # Update uncovered
            for ts_idx in block_cover_indices[best_block]:
                uncovered[ts_idx] -= 1
                
        return selected_blocks

    # 2. Simulated Annealing
    # State: list of block indices
    # Energy: len(blocks)
    # Moves:
    #   - Remove a block if it doesn't make any t-subset uncovered.
    #   - Swap a block with another block that reduces the number of blocks? No, swap keeps count same.
    #   - Replace a block with a different block? Keeps count same.
    #   - The primary move to reduce count is "Remove".
    #   - To escape local minima where we can't remove, we might need to "Swap" to a configuration where removal is possible.
    #   
    # Let's define the state as a set of blocks.
    # Coverage count array: cover_count[ts_idx]
    
    initial_blocks = generate_initial_solution()
    if not initial_blocks:
        # Fallback: all blocks
        initial_blocks = list(range(n_blocks))
        
    # Check if initial is valid
    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
    for b_idx in initial_blocks:
        for ts_idx in block_cover_indices[b_idx]:
            cover_count[ts_idx] += 1
            
    if np.any(cover_count == 0):
        # Need to fix. Add blocks to cover missing.
        # This is a repair step.
        for ts_idx in range(n_tsubsets):
            if cover_count[ts_idx] == 0:
                # Find a block covering it
                candidates = t_subset_to_blocks[ts_idx]
                if candidates:
                    b_idx = candidates[0]
                    initial_blocks.append(b_idx)
                    for ts_idx2 in block_cover_indices[b_idx]:
                        cover_count[ts_idx2] += 1
                        
    best_blocks = initial_blocks.copy()
    best_score = len(best_blocks)
    
    current_blocks = initial_blocks.copy()
    current_score = len(current_blocks)
    
    # Rebuild cover_count for current
    cover_count = np.zeros(n_tsubsets, dtype=np.int32)
    for b_idx in current_blocks:
        for ts_idx in block_cover_indices[b_idx]:
            cover_count[ts_idx] += 1
            
    # Simulated Annealing Loop
    T = 1.0
    T_min = 0.01
    alpha = 0.99
    
    # We need to track which blocks are present for fast removal check.
    # present_blocks = set(current_blocks)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Randomly pick a move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a block
            if len(current_blocks) == 0:
                continue
            b_idx = current_blocks[random.randint(0, len(current_blocks) - 1)]
            
            # Check if any t-subset is uniquely covered by b_idx
            can_remove = True
            for ts_idx in block_cover_indices[b_idx]:
                if cover_count[ts_idx] == 1:
                    can_remove = False
                    break
                    
            if can_remove:
                # Accept
                current_blocks.remove(b_idx)
                for ts_idx in block_cover_indices[b_idx]:
                    cover_count[ts_idx] -= 1
                current_score -= 1
                
                if current_score < best_score:
                    best_score = current_score
                    best_blocks = current_blocks.copy()
            else:
                # Reject (or accept with prob exp(-delta/T) if delta > 0, but here delta is +1 energy)
                # Energy increases by 1 (block count goes up? No, we didn't remove it).
                # If we reject, state stays same.
                pass
                
        elif move_type < 0.9:
            # Try to swap a block with another
            # This keeps the count same, but might allow future removals.
            if len(current_blocks) == 0:
                continue
            old_b_idx = current_blocks[random.randint(0, len(current_blocks) - 1)]
            
            # Pick a new block randomly
            new_b_idx = random.randint(0, n_blocks - 1)
            
            if new_b_idx == old_b_idx:
                continue
                
            # Check if new_b_idx is already in current_blocks
            if new_b_idx in current_blocks:
                continue
                
            # Calculate delta energy.
            # Removing old: cover_count decreases. If any becomes 0, we have a "hole".
            # Adding new: cover_count increases.
            # We want to maintain coverage.
            # If we swap, we must ensure no t-subset becomes uncovered.
            
            # Simulate removal
            holes = []
            for ts_idx in block_cover_indices[old_b_idx]:
                if cover_count[ts_idx] == 1:
                    holes.append(ts_idx)
                    
            if holes:
                # Can we cover these holes with new_b_idx?
                new_covers = set(block_cover_indices[new_b_idx])
                for h in holes:
                    if h not in new_covers:
                        break
                else:
                    # Swap is valid
                    # Update state
                    current_blocks[current_blocks.index(old_b_idx)] = new_b_idx
                    for ts_idx in block_cover_indices[old_b_idx]:
                        cover_count[ts_idx] -= 1
                    for ts_idx in block_cover_indices[new_b_idx]:
                        cover_count[ts_idx] += 1
                    # Score unchanged
                    
            else:
                # No holes, swap is always valid
                current_blocks[current_blocks.index(old_b_idx)] = new_b_idx
                for ts_idx in block_cover_indices[old_b_idx]:
                    cover_count[ts_idx] -= 1
                for ts_idx in block_cover_indices[new_b_idx]:
                    cover_count[ts_idx] += 1
                    
        else:
            # Try to add a block (to escape local minima where we are stuck? 
            # Actually, adding increases score, so we only do this if we think it helps later.
            # Given time constraints, maybe skip this or do it rarely.)
            pass
            
        T *= alpha
        
    # Final cleanup: try to remove any redundant blocks from best_blocks
    # Rebuild cover count for best_blocks
    cover_count_best = np.zeros(n_tsubsets, dtype=np.int32)
    for b_idx in best_blocks:
        for ts_idx in block_cover_indices[b_idx]:
            cover_count_best[ts_idx] += 1
            
    final_blocks = []
    for b_idx in best_blocks:
        needed = False
        for ts_idx in block_cover_indices[b_idx]:
            if cover_count_best[ts_idx] == 1:
                needed = True
                break
        if not needed:
            for ts_idx in block_cover_indices[b_idx]:
                cover_count_best[ts_idx] -= 1
        else:
            final_blocks.append(b_idx)
            
    # Convert back to lists of points
    result = [list(all_blocks[b_idx]) for b_idx in final_blocks]
    
    # Verify
    # (Verification is done by evaluator, but we can double check logic)
    
    return result
