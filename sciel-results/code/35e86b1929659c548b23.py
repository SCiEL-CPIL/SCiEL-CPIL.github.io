import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Pre-compute all t-subsets (targets)
    all_targets = list(itertools.combinations(range(v), t))
    num_targets = len(all_targets)
    
    # Map target to index
    target_to_idx = {tup: i for i, tup in enumerate(all_targets)}
    
    # Pre-compute for each block, the list of target indices it covers
    block_to_targets = []
    for block in all_blocks:
        targets = []
        for t_sub in itertools.combinations(block, t):
            targets.append(target_to_idx[t_sub])
        block_to_targets.append(targets)
    
    # Initialize uncovered targets
    uncovered = set(range(num_targets))
    
    # Greedy Set Cover
    selected_blocks = []
    
    while uncovered:
        if time.time() > deadline - 5.0:
            break
            
        best_gain = 0
        best_block_idx = -1
        
        # Find the block that covers the most uncovered targets
        for i in range(num_blocks):
            # If this block was already selected, skip (optional optimization if we track selected)
            # But in standard greedy, we just pick the best available.
            # We can optimize by not re-evaluating blocks that are already selected if we ensure uniqueness,
            # but simpler is to just pick the max gain.
            # Actually, we must ensure distinct blocks.
            
            # Calculate gain
            # Optimization: if len(uncovered) is small, or if we can break early
            gain = 0
            for t_idx in block_to_targets[i]:
                if t_idx in uncovered:
                    gain += 1
                    # Early termination if gain can't beat best?
                    # Max possible gain is len(uncovered)
                    if gain + (len(block_to_targets[i]) - gain) <= best_gain:
                         # This is not a good early break because we don't know how many more are in uncovered
                         pass
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        
        if best_block_idx == -1 or best_gain == 0:
            # Should not happen if uncovered is not empty
            break
            
        selected_blocks.append(all_blocks[best_block_idx])
        # Remove covered targets from uncovered
        for t_idx in block_to_targets[best_block_idx]:
            uncovered.discard(t_idx)
            
    # Post-processing: Try to remove redundant blocks
    # This can often reduce the number of blocks
    if time.time() < deadline - 10.0:
        # Sort selected blocks by size of their unique contribution (hard to calculate)
        # Simpler: iterate through selected blocks and try to remove each one
        # If removal doesn't uncover any target, keep it removed.
        # We need to check if a block is "essential".
        
        # A block is redundant if all its targets are covered by other selected blocks.
        # We can check this by:
        # 1. For each block i in selected_blocks:
        # 2. Temporarily remove it.
        # 3. Check if any target in block_to_targets[i] is NOT covered by any other block in selected_blocks.
        
        # To make this efficient, we can maintain a count of how many selected blocks cover each target.
        # But selected_blocks changes.
        
        # Let's do a simple pass:
        # For each block in selected_blocks (in reverse order maybe?), check if it's redundant.
        # Redundant if: for all t in block_to_targets[i], there exists j != i such that t in block_to_targets[j].
        
        # This is O(N^2 * T) where N is num blocks, T is targets per block.
        # N ~ 150, T ~ 35. 150*150*35 ~ 800k ops. Very fast.
        
        new_selected = []
        for i, block in enumerate(selected_blocks):
            # Check if this block is redundant given the others in selected_blocks
            # We need to check against all OTHER selected blocks
            is_redundant = True
            for t_idx in block_to_targets[selected_blocks.index(block)]: # Note: index lookup is slow
                 # Better to use indices
                 pass
        
        # Let's use indices properly
        selected_indices = [all_blocks.index(b) for b in selected_blocks]
        
        final_indices = []
        for i in range(len(selected_indices)):
            bi = selected_indices[i]
            # Check if block bi is redundant
            redundant = True
            for t_idx in block_to_targets[bi]:
                # Check if t_idx is covered by any other selected block
                covered_by_other = False
                for j in range(len(selected_indices)):
                    if i == j:
                        continue
                    bj = selected_indices[j]
                    if t_idx in block_to_targets[bj]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    redundant = False
                    break
            
            if not redundant:
                final_indices.append(bi)
        
        # Re-verify the cover with final_indices
        if final_indices:
             covered_targets = set()
             for bi in final_indices:
                 for t_idx in block_to_targets[bi]:
                     covered_targets.add(t_idx)
             if covered_targets == set(range(num_targets)):
                 selected_blocks = [all_blocks[i] for i in final_indices]
             # Else keep original selected_blocks

    return selected_blocks
