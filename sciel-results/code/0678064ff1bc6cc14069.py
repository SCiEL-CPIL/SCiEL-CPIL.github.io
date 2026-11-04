import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Precompute all k-blocks and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute block -> t-subset indices mapping
    k_choose_t = len(list(itertools.combinations(range(k), t)))
    block_subset_indices = np.empty((num_blocks, k_choose_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        for j, subset in enumerate(itertools.combinations(block, t)):
            block_subset_indices[i, j] = t_subset_to_idx[subset]
    
    best_blocks = None
    best_count = num_t_subsets  # worst case upper bound
    
    # Multiple restarts with different random orderings
    num_restarts = 5
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random permutation of blocks for tie-breaking diversity
        block_order = rng.permutation(num_blocks)
        
        # Primal-dual greedy construction
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        dual_vars = np.ones(num_t_subsets, dtype=np.float64)  # Lagrangian multipliers
        
        while uncovered.any():
            if time.time() > deadline - 15:
                break
                
            # Compute weighted gain: sum of dual_vars for uncovered subsets in each block
            # This creates a feedback: uncovered subsets get higher priority
            gains = np.zeros(num_blocks, dtype=np.float64)
            for b_idx in range(num_blocks):
                sub_idx = block_subset_indices[b_idx]
                gains[b_idx] = dual_vars[sub_idx][uncovered[sub_idx]].sum()
            
            # Find best block by gain, break ties using random order
            max_gain = gains.max()
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Use random ordering for tie-breaking
                cand_order = np.array([np.where(block_order == c)[0][0] for c in candidates])
                best_cand = candidates[np.argmin(cand_order)]
            else:
                best_cand = candidates[0]
                
            selected.append(best_cand)
            
            # Update uncovered
            uncovered[block_subset_indices[best_cand]] = False
            
            # Update dual variables: increase for newly covered subsets (feedback)
            # This is a simple subgradient update
            newly_covered = ~uncovered[block_subset_indices[best_cand]]
            # Actually, let's keep duals simple: just track coverage count for redundancy
            pass
        
        if not selected:
            continue
            
        # Convert to set for fast lookup
        selected_set = set(selected)
        
        # Compute coverage counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected:
            cover_count[block_subset_indices[b_idx]] += 1
        
        # Verify coverage
        if cover_count.min() < 1:
            continue
            
        # Redundancy removal (iterative)
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_selected = []
            for b_idx in selected:
                sub_idx = block_subset_indices[b_idx]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Local search: try to swap one block for another that reduces total
        # First, try simple drops again
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            new_selected = []
            for b_idx in selected:
                sub_idx = block_subset_indices[b_idx]
                if np.min(cover_count[sub_idx]) > 1:
                    cover_count[sub_idx] -= 1
                    changed = True
                else:
                    new_selected.append(b_idx)
            selected = new_selected
        
        # Swap-based local search: try replacing pairs
        if time.time() < deadline - 5:
            # Try to find a block that can be replaced by removing it and adding
            # blocks that cover its unique subsets more efficiently
            # Simplified: try random swaps
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                if len(selected) < 2:
                    break
                    # Pick a random block to try removing
                b_remove = random.choice(selected)
                sub_idx_remove = block_subset_indices[b_remove]
                
                # Find subsets only covered by this block
                unique_mask = (cover_count[sub_idx_remove] == 1)
                if not unique_mask.any():
                    # Can just drop
                    cover_count[sub_idx_remove] -= 1
                    selected.remove(b_remove)
                    continue
                    
                # Find blocks that cover these unique subsets
                unique_subsets = sub_idx_remove[unique_mask]
                # Find candidate blocks that cover at least one unique subset
                # and have high overlap with the removed block
                candidates = []
                for b_idx in range(num_blocks):
                    if b_idx == b_remove:
                        continue
                    sub_idx = block_subset_indices[b_idx]
                    # Count how many unique subsets this block covers
                    overlap = np.isin(sub_idx, unique_subsets).sum()
                    if overlap > 0:
                        candidates.append((overlap, b_idx))
                
                if not candidates:
                    continue
                    
                candidates.sort(reverse=True)
                # Try the best candidate
                _, b_add = candidates[0]
                sub_idx_add = block_subset_indices[b_add]
                
                # Check if adding b_add and removing b_remove still covers everything
                # Temporarily remove b_remove
                cover_count[sub_idx_remove] -= 1
                # Add b_add
                cover_count[sub_idx_add] += 1
                
                if cover_count.min() >= 1:
                    # Accept the swap
                    selected.remove(b_remove)
                    if b_add not in selected:
                        selected.append(b_add)
                    else:
                        # b_add was already there, so we just removed b_remove
                        # This is a drop, which is fine
                        pass
                else:
                    # Revert
                    cover_count[sub_idx_remove] += 1
                    cover_count[sub_idx_add] -= 1
        
        # Final redundancy removal
        if time.time() < deadline - 3:
            cover_count = np.zeros(num_t_subsets, dtype=np.int32)
            for b_idx in selected:
                cover_count[block_subset_indices[b_idx]] += 1
                
            changed = True
            while changed and time.time() < deadline - 3:
                changed = False
                new_selected = []
                for b_idx in selected:
                    sub_idx = block_subset_indices[b_idx]
                    if np.min(cover_count[sub_idx]) > 1:
                        cover_count[sub_idx] -= 1
                        changed = True
                    else:
                        new_selected.append(b_idx)
                selected = new_selected
        
        # Check if this is better
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = selected[:]
            
            # Early exit if we've hit a good target
            if best_count <= 270:
                break
    
    # If no valid solution found, fall back to simple greedy
    if best_blocks is None:
        uncovered = np.ones(num_t_subsets, dtype=bool)
        selected = []
        while uncovered.any() and time.time() < deadline - 2:
            gains = uncovered[block_subset_indices].sum(axis=1)
            max_gain_idx = np.argmax(gains)
            if gains[max_gain_idx] == 0:
                break
            selected.append(max_gain_idx)
            uncovered[block_subset_indices[max_gain_idx]] = False
        best_blocks = selected
    
    # Convert indices to actual blocks
    result = [list(all_blocks[i]) for i in best_blocks]
    return result
