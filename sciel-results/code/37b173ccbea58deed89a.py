import itertools
import time
import random
import sys

def construct(v, k, t):
    # Parameters for the task are fixed by the evaluator, but we handle generically
    # However, the bitmask approach relies on the number of t-subsets being manageable.
    # For v=19, k=7, t=4: C(19,4) = 3876. This fits in a Python int.
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for safety
    
    # Precomputation
    # 1. List of all t-subsets and their indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. List of all k-blocks and their bitmasks
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = []
    for block in all_blocks:
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    
    # Helper: Greedy covering starting from a set of uncovered subsets
    # Returns a list of block indices
    def greedy_repair(uncovered_mask, existing_blocks=None):
        if existing_blocks is None:
            existing_blocks = []
        else:
            existing_blocks = list(existing_blocks) # Copy
            
        # If we start with existing blocks, we assume they are valid and 
        # uncovered_mask represents what is NOT covered by them.
        
        while uncovered_mask:
            best_gain = 0
            best_block_idx = -1
            
            # Iterate through all blocks to find the one covering the most uncovered subsets
            # Optimization: We could filter candidates, but scanning 50k is fast enough
            # if we use local variables for speed.
            for i in range(num_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    # If gain is 0, no block can cover it (should not happen if valid)
                    # If gain is small, we might break early if we find a 'perfect' block?
                    # No, we need the max.
            
            if best_block_idx == -1:
                # Should not happen if the problem is solvable
                break
                
            existing_blocks.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return existing_blocks

    # Helper: Remove redundant blocks
    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        # Count coverage for each t-subset
        cover_count = [0] * num_t_subsets
        
        # Pre-compute the list of t-subset indices for each block to speed up counting
        # We can reuse the generation or store it. Storing is better.
        # Let's create a list of lists: block_subset_indices[i] = list of t-indices in block i
        # This takes memory but speeds up reduction.
        # C(7,4) = 35. 50k * 35 = 1.75M integers. ~14MB. Acceptable.
        
        # Actually, we can just re-iterate combinations in the loop if reduction is infrequent.
        # But reduction is called in every ILS iteration. So precompute.
        
        # Let's precompute block_subset_indices globally
        # Wait, I'll define it outside the function or cache it.
        
        for idx in block_indices:
            # Get the t-subsets of this block
            # We need the actual points to generate combinations?
            # Yes, block_indices are indices into all_blocks.
            block_points = all_blocks[idx]
            for sub in itertools.combinations(block_points, t):
                cover_count[t_subset_to_idx[sub]] += 1
                
        new_blocks = []
        for idx in block_indices:
            block_points = all_blocks[idx]
            # Subtract coverage
            subs_to_check = []
            for sub in itertools.combinations(block_points, t):
                s_idx = t_subset_to_idx[sub]
                cover_count[s_idx] -= 1
                subs_to_check.append(s_idx)
                
            # Check if any count is 0
            is_redundant = True
            for s_idx in subs_to_check:
                if cover_count[s_idx] == 0:
                    is_redundant = False
                    break
                    
            if is_redundant:
                # Keep removed
                pass
            else:
                # Restore coverage
                for s_idx in subs_to_check:
                    cover_count[s_idx] += 1
                new_blocks.append(idx)
                
        return new_blocks

    # Initial Solution
    full_mask = (1 << num_t_subsets) - 1
    initial_blocks = greedy_repair(full_mask)
    current_blocks = reduce_blocks(initial_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    # ILS Loop
    while time.time() < deadline:
        # Perturb
        # Pick a random block to remove
        if not current_blocks:
            break
            
        # To avoid removing the only block covering a subset immediately (which forces a big repair),
        # we just pick random. The repair will handle it.
        
        # Create a copy
        perturbed_blocks = list(current_blocks)
        
        # Remove a random block
        remove_idx = random.randint(0, len(perturbed_blocks) - 1)
        removed_block = perturbed_blocks[remove_idx]
        perturbed_blocks.pop(remove_idx)
        
        # Calculate uncovered mask
        # We need to know which subsets were covered ONLY by the removed block.
        # Easier: Recalculate uncovered mask from scratch?
        # Or maintain cover counts?
        # Recalculating from scratch is O(N * 35). N=150. 5250 ops. Very fast.
        
        covered_mask = 0
        for idx in perturbed_blocks:
            covered_mask |= block_masks[idx]
            
        uncovered_mask = full_mask & ~covered_mask
        
        # Repair
        repaired_blocks = greedy_repair(uncovered_mask, perturbed_blocks)
        
        # Reduce
        new_blocks = reduce_blocks(repaired_blocks)
        
        # Acceptance
        new_size = len(new_blocks)
        
        if new_size < best_size:
            best_blocks = new_blocks
            best_size = new_size
            current_blocks = new_blocks
        else:
            # Accept with some probability to escape local optima
            # Simple random walk: accept if better or equal, else with prob 0.5
            if new_size == best_size:
                # Keep it to explore
                if random.random() < 0.5:
                    current_blocks = new_blocks
            else:
                # Worse
                if random.random() < 0.3: # Lower probability for worse
                    current_blocks = new_blocks
                else:
                    # Revert to best or previous?
                    # Reverting to best prevents drift, but might get stuck.
                    # Let's try reverting to best if we are stuck for a while?
                    # For now, just revert to best to stay anchored.
                    current_blocks = best_blocks
                    # Or keep current_blocks as the one we just tried?
                    # If we always revert to best, we only explore neighbors of best.
                    # This is "Best-First" search.
                    # Let's try keeping the worse solution sometimes to jump over valleys.
                    # But "Best-First" is safer for time limits.
                    # Let's stick to: If worse, revert to best.
                    # This ensures we don't waste time on bad paths.
                    pass

    # Convert indices back to lists of points
    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    
    # Final validation (optional, but good for safety)
    # Check if all t-subsets are covered
    # Given the logic, it should be, but let's be sure.
    # If invalid, return the initial greedy solution (which is guaranteed valid)
    # Actually, if ILS breaks validity, we return best_blocks which was validated by reduce/greedy.
    # Greedy always produces valid. Reduce preserves validity.
    
    return final_blocks

# Pre-computation for speed in reduce_blocks
# We need block_subset_indices to avoid regenerating combinations in reduce_blocks
# Let's refactor to precompute this once.

# Since construct is called once, we can do this inside.

# To optimize reduce_blocks, we will precompute the t-subset indices for each block.
# This avoids calling itertools.combinations in the inner loop of reduce_blocks.

# Revised construct with precomputation:

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute masks and subset indices for each block
    block_masks = []
    block_subset_indices = []
    
    for block in all_blocks:
        mask = 0
        subs_idx = []
        for sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[sub]
            mask |= (1 << idx)
            subs_idx.append(idx)
        block_masks.append(mask)
        block_subset_indices.append(subs_idx)
        
    def greedy_repair(uncovered_mask, existing_blocks=None):
        if existing_blocks is None:
            existing_blocks = []
        else:
            existing_blocks = list(existing_blocks)
            
        while uncovered_mask:
            best_gain = 0
            best_block_idx = -1
            
            for i in range(num_blocks):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                break
                
            existing_blocks.append(best_block_idx)
            uncovered_mask &= ~block_masks[best_block_idx]
            
        return existing_blocks

    def reduce_blocks(block_indices):
        if not block_indices:
            return []
            
        cover_count = [0] * num_t_subsets
        
        for idx in block_indices:
            for s_idx in block_subset_indices[idx]:
                cover_count[s_idx] += 1
                
        new_blocks = []
        for idx in block_indices:
            subs = block_subset_indices[idx]
            for s_idx in subs:
                cover_count[s_idx] -= 1
                
            is_redundant = True
            for s_idx in subs:
                if cover_count[s_idx] == 0:
                    is_redundant = False
                    break
                    
            if is_redundant:
                pass
            else:
                for s_idx in subs:
                    cover_count[s_idx] += 1
                new_blocks.append(idx)
                
        return new_blocks

    full_mask = (1 << num_t_subsets) - 1
    initial_blocks = greedy_repair(full_mask)
    current_blocks = reduce_blocks(initial_blocks)
    best_blocks = list(current_blocks)
    best_size = len(best_blocks)
    
    while time.time() < deadline:
        if not current_blocks:
            break
            
        perturbed_blocks = list(current_blocks)
        remove_idx = random.randint(0, len(perturbed_blocks) - 1)
        perturbed_blocks.pop(remove_idx)
        
        covered_mask = 0
        for idx in perturbed_blocks:
            covered_mask |= block_masks[idx]
            
        uncovered_mask = full_mask & ~covered_mask
        
        repaired_blocks = greedy_repair(uncovered_mask, perturbed_blocks)
        new_blocks = reduce_blocks(repaired_blocks)
        
        new_size = len(new_blocks)
        
        if new_size < best_size:
            best_blocks = new_blocks
            best_size = new_size
            current_blocks = new_blocks
        else:
            if new_size == best_size:
                if random.random() < 0.5:
                    current_blocks = new_blocks
            else:
                if random.random() < 0.2:
                    current_blocks = new_blocks
                else:
                    current_blocks = best_blocks

    final_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return final_blocks
