import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    
    # Map t-subsets to indices for fast array access
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (candidates)
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    # Precompute the t-subsets contained in each k-subset
    # This is a list of lists of indices
    k_to_t_indices = []
    for k_sub in all_k_subsets:
        t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            t_indices.append(t_subset_to_idx[t_sub])
        k_to_t_indices.append(t_indices)
        
    # Greedy Set Cover
    covered_count = [0] * n_t
    selected_blocks = []
    
    # We need to find the block with max gain
    # Optimization: We can break early if we have covered all
    remaining = n_t
    
    for _ in range(n_t): # Max possible blocks
        if time.time() - start_time > timeout:
            break
            
        best_gain = 0
        best_block_idx = -1
        
        # Iterate through all candidates
        # We can skip blocks that are subsets of already selected blocks? 
        # No, greedy set cover doesn't do that easily without complex checks.
        
        for i, t_indices in enumerate(k_to_t_indices):
            # Calculate gain: how many of these t_subsets have count 0?
            gain = 0
            for idx in t_indices:
                if covered_count[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
                if gain == len(t_indices): # Perfect block, can't do better
                    # Note: We still need to check if other blocks have same gain? 
                    # No, we just want *a* block with max gain.
                    # But we should still finish the loop? No, if gain is max possible for this block, 
                    # and it's the max possible overall? 
                    # Actually, max possible gain is len(t_indices). 
                    # If we find one, it's the best we can do for this step.
                    # We can break inner loop? No, we are in the outer loop.
                    # We can break the outer loop if we found a block with gain == len(t_indices) 
                    # AND we assume no other block can have higher gain? 
                    # Yes, gain cannot exceed len(t_indices).
                    # So if best_gain == len(t_indices), it's the global max.
                    pass 
        
        if best_gain == 0:
            # Should not happen if a covering exists
            break
            
        # Select the best block
        selected_blocks.append(all_k_subsets[best_block_idx])
        
        # Update covered counts
        for idx in k_to_t_indices[best_block_idx]:
            if covered_count[idx] == 0:
                remaining -= 1
            covered_count[idx] += 1
            
        if remaining == 0:
            break
            
    # Minimization: Remove redundant blocks
    # We iterate through selected_blocks and try to remove them
    # We need to re-evaluate coverage from scratch or maintain state?
    # Maintaining state is tricky if we remove a block, counts change.
    # Simpler: Re-calculate coverage for the current set of blocks, 
    # then try removing each one.
    
    # Let's do a pass of minimization
    # This might be slow if done naively on the full set repeatedly.
    # But N_blocks is ~150. N_t is 3876.
    # Checking if a block is redundant:
    # 1. Compute coverage of all blocks EXCEPT this one.
    # 2. Check if any t_subset has count 0.
    
    # More efficient minimization:
    # Maintain `covered_count`.
    # For each block b in selected_blocks:
    #   Temporarily subtract b's contribution from covered_count.
    #   Check if min(covered_count) > 0.
    #   If yes, keep it removed.
    #   If no, restore b's contribution.
    
    # However, removing a block changes the "essentiality" of others.
    # So we should iterate until no more blocks can be removed.
    
    # Let's implement a simple iterative minimization
    # We'll use a list for current blocks
    current_blocks = list(selected_blocks)
    
    changed = True
    while changed and (time.time() - start_time < timeout - 5):
        changed = False
        # Recalculate covered_count from scratch for current_blocks
        covered_count = [0] * n_t
        for block in current_blocks:
            # Find index of block? Or just iterate t-subsets of block
            for t_sub in itertools.combinations(block, t):
                covered_count[t_subset_to_idx[t_sub]] += 1
                
        # Try to remove each block
        for i in range(len(current_blocks)):
            if time.time() - start_time > timeout - 5:
                break
            block_to_try = current_blocks[i]
            # Subtract its contribution
            for t_sub in itertools.combinations(block_to_try, t):
                covered_count[t_subset_to_idx[t_sub]] -= 1
                
            # Check if any count is 0
            is_redundant = True
            for count in covered_count:
                if count == 0:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Keep it removed
                current_blocks.pop(i)
                changed = True
                # Since we popped, indices shift, restart loop
                break
            else:
                # Restore contribution
                for t_sub in itertools.combinations(block_to_try, t):
                    covered_count[t_subset_to_idx[t_sub]] += 1

    return [list(b) for b in current_blocks]
