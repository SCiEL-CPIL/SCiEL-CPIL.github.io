import itertools
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_indices = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t = len(all_t_subsets)
    
    # Precompute list of t-subsets for each block? 
    # C(24,7) is 346k blocks. Storing the t-subsets for each is heavy.
    # Instead, we compute on the fly or use bitmasks if possible.
    # For v=24, we can use bitmasks for blocks and t-subsets.
    
    # Bitmask representation
    # t-subset mask
    t_masks = [0] * num_t
    for i, ts in enumerate(all_t_subsets):
        mask = 0
        for x in ts:
            mask |= (1 << x)
        t_masks[i] = mask
        
    # Helper to get t-subset indices in a block
    # This is called frequently.
    # C(7,3) = 35 t-subsets per block.
    
    def get_block_t_indices(block):
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_subset_indices[ts])
        return indices

    def greedy_cover():
        covered = [0] * num_t
        blocks = []
        uncovered_indices = list(range(num_t))
        
        while uncovered_indices:
            # Pick a random uncovered t-subset to focus on
            idx = random.choice(uncovered_indices)
            target_t = all_t_subsets[idx]
            
            # Find best block containing target_t
            remaining_points = [x for x in range(v) if x not in target_t]
            best_block = None
            best_gain = -1
            
            # Iterate over all k-subsets containing target_t
            # C(21, 4) = 5985 candidates
            for ext in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_t + ext))
                # Calculate gain
                gain = 0
                for ts in itertools.combinations(block, t):
                    t_idx = t_subset_indices[ts]
                    if covered[t_idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if gain == len(uncovered_indices): # Perfect cover of all remaining? Unlikely but check
                        break
            
            blocks.append(list(best_block))
            
            # Update covered
            for ts in itertools.combinations(best_block, t):
                t_idx = t_subset_indices[ts]
                if covered[t_idx] == 0:
                    covered[t_idx] = 1
            
            # Rebuild uncovered list
            uncovered_indices = [i for i in uncovered_indices if covered[i] == 0]
            
        return blocks

    def prune(blocks):
        """Remove redundant blocks. Returns new list of blocks."""
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        counts = [0] * num_t
        block_t_indices = []
        for block in blocks:
            t_inds = get_block_t_indices(block)
            block_t_indices.append(t_inds)
            for idx in t_inds:
                counts[idx] += 1
                
        essential_blocks = []
        for i, block in enumerate(blocks):
            is_essential = False
            for idx in block_t_indices[i]:
                if counts[idx] == 1:
                    is_essential = True
                    break
            if is_essential:
                essential_blocks.append(block)
                
        return essential_blocks

    def find_best_replacement(uncovered_indices):
        """
        Given a list of uncovered t-subset indices, find a k-block that covers the most of them.
        Uses a focal point heuristic: pick a few random t-subsets from uncovered,
        and search for best k-block containing that t-subset.
        """
        if not uncovered_indices:
            return None, 0
            
        best_block = None
        best_gain = 0
        
        # Try a few focal points
        # If uncovered is small, try all. If large, sample.
        focal_points = uncovered_indices if len(uncovered_indices) < 10 else random.sample(uncovered_indices, 10)
        
        for f_idx in focal_points:
            target_t = all_t_subsets[f_idx]
            remaining_points = [x for x in range(v) if x not in target_t]
            
            for ext in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_t + ext))
                # Calculate gain against uncovered_indices
                # Optimization: Only check if it covers the focal point (it does by construction)
                # and count how many other uncovered ones it covers.
                gain = 1 # At least covers focal point
                # We need to check all t-subsets in this block against the uncovered set
                # To make this fast, convert uncovered_indices to a set
                # But creating a set every time is expensive.
                # Alternative: Precompute a bitmask of uncovered? 
                # Or just iterate. 35 t-subsets per block.
                
                # Let's just iterate
                block_t_set = set(itertools.combinations(block, t))
                # This is slow.
                
                # Better:
                # For each ts in block, check if its index is in uncovered_indices.
                # We can use a boolean array for uncovered status for this call?
                # No, uncovered_indices is a list.
                
                # Let's optimize the gain calculation.
                # We know target_t is in block and uncovered.
                # We need to count other ts in block that are in uncovered_indices.
                
                # To avoid set creation, we can just check membership in a set built once per replacement call.
                # But we are inside a loop over blocks.
                
                # Let's assume the list of uncovered is small enough or we build a set outside.
                pass # Placeholder for logic structure

        # The logic above needs to be tighter. 
        # Let's refactor find_best_replacement to be efficient.
        pass

    # Refactored find_best_replacement
    def find_best_replacement_efficient(uncovered_indices):
        if not uncovered_indices:
            return None, 0
            
        # Create a set for fast lookup of uncovered indices
        uncovered_set = set(uncovered_indices)
        
        best_block = None
        best_gain = 0
        
        # Sample focal points
        num_focals = min(len(uncovered_indices), 5)
        focal_points = random.sample(uncovered_indices, num_focals)
        
        for f_idx in focal_points:
            target_t = all_t_subsets[f_idx]
            remaining_points = [x for x in range(v) if x not in target_t]
            
            for ext in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(target_t + ext))
                
                gain = 0
                # Check all t-subsets in this block
                for ts in itertools.combinations(block, t):
                    if t_subset_indices[ts] in uncovered_set:
                        gain += 1
                        
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
                    if gain == len(uncovered_indices):
                        return block, gain # Perfect fix found early?
                        
        return best_block, best_gain

    # Local Search
    start_time = time.time()
    
    # 1. Initial Solution
    current_blocks = greedy_cover()
    current_blocks = prune(current_blocks)
    best_blocks = current_blocks
    best_len = len(current_blocks)
    
    # 2. Local Search Loop
    # Strategy: 
    # - Pick a random block to remove.
    # - If it causes uncovered sets, try to fix with one replacement block.
    # - Prune.
    # - If size improves, accept.
    # - If size stays same, accept with probability 0.5 to explore.
    
    while time.time() - start_time < 110:
        if len(current_blocks) < 2:
            break
            
        # Pick a block to remove
        remove_idx = random.randint(0, len(current_blocks) - 1)
        remove_block = current_blocks[remove_idx]
        
        # Calculate uncovered sets if we remove this block
        # We need to know which t-subsets are ONLY covered by this block.
        # We need coverage counts.
        # To avoid recalculating full counts every step, we can maintain them, 
        # but for simplicity and correctness with random swaps, let's recalculate 
        # or use a faster check.
        
        # Fast check:
        # Count how many blocks cover each t-subset in remove_block.
        # If count == 1 for any, those become uncovered.
        
        # This is O(num_blocks * C(k,t)). 76 * 35 ~ 2600 ops. Very fast.
        counts = [0] * num_t
        for block in current_blocks:
            for ts in itertools.combinations(block, t):
                counts[t_subset_indices[ts]] += 1
                
        uncovered_indices = []
        for ts in itertools.combinations(remove_block, t):
            idx = t_subset_indices[ts]
            if counts[idx] == 1:
                uncovered_indices.append(idx)
                
        # Create new list without remove_block
        new_blocks = [b for i, b in enumerate(current_blocks) if i != remove_idx]
        
        if not uncovered_indices:
            # Safe removal
            new_blocks = prune(new_blocks)
            if len(new_blocks) < best_len:
                best_blocks = new_blocks
                best_len = len(new_blocks)
            current_blocks = new_blocks
            continue
            
        # Try to fix uncovered_indices with one block
        replacement_block, gain = find_best_replacement_efficient(uncovered_indices)
        
        if replacement_block is None:
            continue
            
        # Add replacement
        new_blocks.append(list(replacement_block))
        
        # Prune
        new_blocks = prune(new_blocks)
        
        new_len = len(new_blocks)
        
        if new_len < len(current_blocks):
            # Improvement
            current_blocks = new_blocks
            if new_len < best_len:
                best_blocks = current_blocks
                best_len = new_len
        elif new_len == len(current_blocks):
            # Neutral move, accept to explore
            if random.random() < 0.5:
                current_blocks = new_blocks
                if new_len < best_len:
                    best_blocks = current_blocks
                    best_len = new_len
        # Else: worse, reject

    return best_blocks
