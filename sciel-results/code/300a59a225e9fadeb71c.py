import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety

    # Precompute all 6-subsets and their 5-subset members
    all_blocks = list(combinations(range(12), 6))
    block_5subsets = [list(combinations(b, 5)) for b in all_blocks]
    
    # Map each 5-subset to an index for fast lookup
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    n_5subsets = len(all_5subsets)  # 792
    
    # Precompute bitmask representation for each block's 5-subsets
    block_masks = []
    for b in all_blocks:
        mask = 0
        for s in combinations(b, 5):
            mask |= (1 << subset_to_idx[s])
        block_masks.append(mask)
    
    # Greedy: pick blocks in lexicographic order, check via bitmask
    covered_mask = 0
    blocks = []
    
    for i, b in enumerate(all_blocks):
        if time.time() > deadline:
            break
        if (covered_mask & block_masks[i]) == 0:
            blocks.append(list(b))
            covered_mask |= block_masks[i]
            if len(blocks) == 132:
                return blocks
    
    # Local search to improve the solution
    if len(blocks) < 132:
        best_blocks = blocks
        best_count = len(blocks)
        
        for trial in range(100):
            if time.time() > deadline:
                break
            random.seed(trial)
            
            # Start with a copy of the current best solution
            current_blocks = [list(b) for b in best_blocks]
            current_mask = 0
            for b in current_blocks:
                bi = all_blocks.index(tuple(b))
                current_mask |= block_masks[bi]
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                
                # Try to add a new block
                for i, b in enumerate(all_blocks):
                    if (current_mask & block_masks[i]) == 0:
                        current_blocks.append(list(b))
                        current_mask |= block_masks[i]
                        improved = True
                        if len(current_blocks) == 132:
                            return current_blocks
                        break
                
                # If no new block can be added, try swapping
                if not improved:
                    for i, b in enumerate(all_blocks):
                        if (current_mask & block_masks[i]) != 0:
                            continue
                        
                        # Try swapping each existing block with this new block
                        for j, existing_block in enumerate(current_blocks):
                            existing_idx = all_blocks.index(tuple(existing_block))
                            new_mask = current_mask & ~block_masks[existing_idx] | block_masks[i]
                            
                            # Check if the new mask is valid (no conflicts)
                            if new_mask == current_mask & ~block_masks[existing_idx] | block_masks[i]:
                                # Swap the blocks
                                current_blocks[j] = list(b)
                                current_mask = new_mask
                                improved = True
                                break
                        
                        if improved:
                            break
            
            if len(current_blocks) > best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks
                if best_count == 132:
                    return best_blocks
    
    return best_blocks
