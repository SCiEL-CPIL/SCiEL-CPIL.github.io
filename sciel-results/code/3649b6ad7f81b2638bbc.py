import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Precompute all t-subsets as bitmasks
    points = range(v)
    all_t_subsets = []
    subset_to_idx = {}
    for i, comb in enumerate(itertools.combinations(points, t)):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        all_t_subsets.append(mask)
        subset_to_idx[mask] = i
        
    n_subsets = len(all_t_subsets)
    
    # Helper to get t-subsets of a block (list of points)
    def get_block_t_subsets(block):
        masks = []
        for comb in itertools.combinations(block, t):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            masks.append(mask)
        return masks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple attempts
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        # Randomize the order of uncovered sets to break symmetry
        # We represent uncovered as a set of indices or masks?
        # Masks are easier for bit operations.
        uncovered = set(all_t_subsets)
        blocks = []
        
        # Greedy construction
        while uncovered:
            # Pick the first uncovered subset (or random)
            # Using min is deterministic, let's use random for multi-start
            if attempt == 1:
                current_t = min(uncovered)
            else:
                current_t = random.choice(list(uncovered))
                
            block = []
            # Initialize block with points from current_t
            # current_t is a bitmask
            for p in points:
                if current_t & (1 << p):
                    block.append(p)
            
            # Add points to block until size k
            while len(block) < k:
                best_gain = -1
                best_point = -1
                block_mask = 0
                for p in block:
                    block_mask |= (1 << p)
                
                # Try adding each point not in block
                for p in points:
                    if block_mask & (1 << p):
                        continue
                    new_mask = block_mask | (1 << p)
                    gain = 0
                    # Count how many uncovered subsets contain new_mask
                    for t_mask in uncovered:
                        if (t_mask & new_mask) == new_mask:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                        if gain == len(uncovered): # Max possible
                            break
                
                if best_point == -1:
                    # Should not happen if uncovered is not empty
                    break
                block.append(best_point)
            
            # Add block
            block.sort()
            blocks.append(block)
            
            # Remove covered subsets
            block_t_subsets = get_block_t_subsets(block)
            for mask in block_t_subsets:
                uncovered.discard(mask)
        
        # Reduction phase
        # Count coverage
        cover_count = {}
        for mask in all_t_subsets:
            cover_count[mask] = 0
        
        for block in blocks:
            for mask in get_block_t_subsets(block):
                cover_count[mask] += 1
        
        # Try to remove blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for block in blocks:
                block_masks = get_block_t_subsets(block)
                # Check if any mask in this block has count 1
                essential = False
                for mask in block_masks:
                    if cover_count[mask] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove block
                    for mask in block_masks:
                        cover_count[mask] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
            
            if time.time() > deadline:
                break
        
        # Update best
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = [b[:] for b in blocks]
            
        # If we reached the theoretical lower bound or a very good number, stop early?
        # Lower bound is 264. Best known 335.
        # If we get below 335, keep trying to go lower.
        
    return best_blocks if best_blocks is not None else []
