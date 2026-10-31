import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Precompute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to the list of t-subset indices it covers
    block_to_tsubs = []
    for b in all_blocks:
        indices = [t_sub_to_idx[ts] for ts in itertools.combinations(b, t)]
        block_to_tsubs.append(indices)
        
    # Reverse map: t_sub_idx -> list of block indices
    t_sub_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_to_tsubs):
        for t_idx in t_indices:
            t_sub_to_blocks[t_idx].append(b_idx)
            
    # Convert to sets for faster intersection if needed, or keep lists
    # For v=19, k=7, t=4, each block covers 35 t-subsets.
    # Each t-subset is in C(15, 3) = 455 blocks.
    
    cover_count = np.zeros(num_t_subsets, dtype=np.int32)
    blocks_in_cover = []
    
    # Helper to update cover counts
    def add_block(b_idx):
        for idx in block_to_tsubs[b_idx]:
            cover_count[idx] += 1
            
    def remove_block(b_idx):
        for idx in block_to_tsubs[b_idx]:
            cover_count[idx] -= 1

    # 1. Greedy Initialization
    # Global Greedy: Pick the block that covers the most uncovered t-subsets.
    # To speed this up, we can maintain a list of uncovered t-subsets.
    # However, calculating gain for all 50k blocks every step is expensive.
    # Alternative: Pick a random uncovered t-subset, find best block for it.
    # This is often faster and nearly as good.
    
    uncovered_mask = np.ones(num_t_subsets, dtype=bool)
    
    while np.any(uncovered_mask):
        if time.time() > deadline - 5: break
        
        # Pick a random uncovered t-subset
        uncovered_indices = np.where(uncovered_mask)[0]
        target_idx = random.choice(uncovered_indices)
        
        # Candidates are blocks covering this target
        candidates = t_sub_to_blocks[target_idx]
        
        best_block_idx = -1
        max_gain = -1
        
        # Evaluate candidates
        # Gain = number of t-subsets in block with cover_count == 0
        for b_idx in candidates:
            gain = 0
            for t_idx in block_to_tsubs[b_idx]:
                if cover_count[t_idx] == 0:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx != -1:
            blocks_in_cover.append(best_block_idx)
            add_block(best_block_idx)
            # Update uncovered mask
            for t_idx in block_to_tsubs[best_block_idx]:
                if cover_count[t_idx] == 1:
                    uncovered_mask[t_idx] = False

    # 2. Local Search
    # Strategy: Targeted Swaps based on Critical Subsets
    
    # Precompute sets for block t-subsets for faster membership/intersection
    block_tsub_sets = [set(indices) for indices in block_to_tsubs]
    
    # Helper to get unique t-subsets for a block (count == 1)
    def get_unique_tsubs(b_idx):
        return [t_idx for t_idx in block_to_tsubs[b_idx] if cover_count[t_idx] == 1]
    
    # Helper to check if a block is redundant
    def is_redundant(b_idx):
        for t_idx in block_to_tsubs[b_idx]:
            if cover_count[t_idx] == 1:
                return False
        return True

    # Main Local Search Loop
    while time.time() < deadline - 1:
        improved = False
        
        # Phase 1: Deletion
        # Check if any block is redundant
        for b_idx in list(blocks_in_cover):
            if is_redundant(b_idx):
                remove_block(b_idx)
                blocks_in_cover.remove(b_idx)
                improved = True
        
        if improved:
            continue # Restart as state changed
            
        # Phase 2: Targeted Swap
        # Find a block that has unique t-subsets (critical block)
        # We want to swap it with a block that covers all its unique t-subsets.
        
        # To avoid checking all blocks every time, we can iterate through blocks_in_cover
        # and try to find a replacement.
        
        # Optimization: Only consider blocks that have at least one unique t-subset?
        # No, swapping a non-critical block might allow removing another block later?
        # Usually, we swap a block that is "wasting" space (has unique parts) with one that covers those parts better.
        
        # Let's pick a random block from the cover that has unique t-subsets.
        # If no block has unique t-subsets, the cover is "tight" (no block can be removed or swapped to reduce count directly without breaking coverage, unless we do 2-swaps).
        
        critical_blocks = []
        for b_idx in blocks_in_cover:
            if any(cover_count[t_idx] == 1 for t_idx in block_to_tsubs[b_idx]):
                critical_blocks.append(b_idx)
                
        if not critical_blocks:
            # No single block has unique subsets. 
            # This implies every subset is covered by >= 2 blocks.
            # We might still be able to swap to create redundancy?
            # If every subset is covered by >= 2, we can try to remove a block?
            # If we remove a block, its subsets drop to count >= 1. 
            # So if we remove it, coverage is still valid?
            # Wait, if count >= 2, removing one makes count >= 1. So it IS redundant.
            # So if critical_blocks is empty, we should have found a redundant block in Phase 1.
            # Unless... my is_redundant logic is correct. 
            # If no block has a unique subset, then for every block, all its subsets have count >= 2.
            # Then every block is redundant.
            # So Phase 1 should have removed them.
            # If Phase 1 didn't remove them, maybe time ran out?
            # Or maybe I should break to avoid infinite loop if no improvement.
            break
            
        # Pick a random critical block
        b_old_idx = random.choice(critical_blocks)
        unique_tsubs = get_unique_tsubs(b_old_idx)
        
        # We need a block b_new_idx that covers ALL t_idx in unique_tsubs.
        # Find intersection of t_sub_to_blocks[t_idx] for all t_idx in unique_tsubs.
        
        # Start with the list for the first unique t-subset
        if not unique_tsubs:
            continue
            
        candidate_blocks = set(t_sub_to_blocks[unique_tsubs[0]])
        
        for t_idx in unique_tsubs[1:]:
            candidate_blocks.intersection_update(t_sub_to_blocks[t_idx])
            if not candidate_blocks:
                break
        
        if candidate_blocks:
            # Filter out b_old_idx itself? 
            # If b_old_idx is in candidate_blocks, it means it covers its own unique subsets (obviously).
            # We want a DIFFERENT block.
            candidate_blocks.discard(b_old_idx)
            
            if candidate_blocks:
                # Pick a random candidate
                # Prefer candidates not in blocks_in_cover?
                # If we pick one in blocks_in_cover, we are effectively just removing b_old_idx 
                # (since b_new is already there). This is a valid move (deletion via swap).
                # If we pick one NOT in blocks_in_cover, we are swapping.
                
                # Let's pick randomly from candidates.
                b_new_idx = random.choice(list(candidate_blocks))
                
                # Perform swap
                remove_block(b_old_idx)
                add_block(b_new_idx)
                
                blocks_in_cover.remove(b_old_idx)
                if b_new_idx not in blocks_in_cover:
                    blocks_in_cover.append(b_new_idx)
                
                improved = True
                # Continue to check for more improvements in this pass?
                # The 'continue' at the top of the while loop restarts the whole process.
                # This is safer.
                continue
        
        # If we get here, no swap was found for the chosen critical block.
        # To avoid getting stuck, we could try other critical blocks, but for simplicity 
        # and speed, we just loop. If we never find a swap, we might loop forever.
        # Add a counter or break condition?
        # For now, if we don't find a swap in a full pass over critical blocks, we should break.
        # But we only picked one random one.
        # Let's do a full pass over critical blocks in one iteration of the while loop?
        # That might be too slow.
        
        # Alternative: If no swap found for this random pick, just continue. 
        # The randomness will eventually find a swap or we hit the deadline.
        
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in blocks_in_cover]
    
    # Sort blocks for deterministic output
    final_blocks.sort()
    return final_blocks
