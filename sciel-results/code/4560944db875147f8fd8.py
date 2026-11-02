import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # 1. Precomputation
    # Generate all t-subsets and map them to IDs
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {tsub: i for i, tsub in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, precompute the list of t-subset IDs it covers
    # Shape: (n_blocks, t_subsets_per_block)
    t_subsets_per_block = np.zeros((n_blocks, 0), dtype=int)
    block_tsubs_list = []
    for b in all_blocks:
        ids = [tsub_to_id[ts] for ts in itertools.combinations(b, t)]
        block_tsubs_list.append(ids)
        
    # Convert to numpy array for fast vectorized operations
    # All blocks cover the same number of t-subsets: C(k, t)
    # So we can make a 2D array
    block_tsubs = np.array(block_tsubs_list, dtype=int)
    
    # 2. Greedy Construction
    # uncovered: boolean array of length n_tsubs
    uncovered = np.ones(n_tsubs, dtype=bool)
    chosen_block_indices = []
    
    while np.any(uncovered):
        if time.time() - start_time > timeout:
            break
            
        # Calculate score for each block: number of uncovered t-subsets it covers
        # uncovered[block_tsubs] creates a boolean array of shape (n_blocks, C(k,t))
        # summing along axis 1 gives the count
        scores = uncovered[block_tsubs].sum(axis=1)
        
        # Find the block with the maximum score
        best_idx = np.argmax(scores)
        
        # If the best score is 0, we are stuck (should not happen if uncovered is not empty)
        if scores[best_idx] == 0:
             # Fallback: pick any block? Or break?
             # If uncovered is not empty, some block must cover at least 1.
             # This implies a bug or logic error. 
             # For safety, just break to avoid infinite loop.
             break
             
        chosen_block_indices.append(best_idx)
        
        # Update uncovered
        uncovered[block_tsubs[best_idx]] = False
        
    # 3. Local Search: Block Removal
    # We have a list of chosen block indices.
    # We want to remove any block that is not essential.
    # A block is essential if it is the *only* block covering some t-subset.
    
    if not chosen_block_indices:
        return [list(all_blocks[i]) for i in chosen_block_indices]
        
    # Calculate coverage counts
    cover_counts = np.zeros(n_tsubs, dtype=int)
    for idx in chosen_block_indices:
        cover_counts[block_tsubs[idx]] += 1
        
    # Greedy Removal
    # We iterate through the chosen blocks.
    # To make removal effective, we might want to remove blocks that are "most redundant" first?
    # Or just iterate in order.
    # Since removing a block reduces counts, it might make other blocks essential.
    # So we should check for essentiality *before* removing.
    # If a block is redundant (min count > 1), we remove it.
    
    # To be efficient, we can loop until no more removals are possible.
    # But typically one pass is enough for greedy outputs.
    # Let's do a while loop to be safe, but it should converge quickly.
    
    current_blocks = list(chosen_block_indices)
    removed = False
    
    # Sort blocks by "redundancy" (max count)? No, just iterate.
    # We need to be careful: if we remove B1, B2 might become essential.
    # So we must check B2 *after* B1 is removed?
    # Actually, if B2 was redundant *before* B1 was removed, it might still be redundant or become essential.
    # If B2 becomes essential, we keep it.
    # So a single pass where we check `min(cover_counts[block_tsubs[idx]]) > 1` is safe.
    # If it is > 1, we remove it and update counts.
    # If it is == 1, we keep it.
    
    # However, if we process in arbitrary order, we might miss a block that becomes redundant *after* another is removed?
    # No. Removing blocks only *decreases* counts.
    # If `min(counts) > 1`, it is redundant.
    # If we remove it, counts drop.
    # Other blocks might have `min(counts) == 1` now, so they are essential.
    # They were essential before too (since counts were higher, if min was 1 before, it's 1 now? No, if min was 2 before, it could be 1 now).
    # So blocks that were essential stay essential. Blocks that were redundant *might* become essential.
    # Therefore, we never "create" a new redundant block by removing others.
    # We only "lose" redundancy.
    # So a single pass is sufficient to find a minimal cover?
    # Not necessarily minimal, but a "greedy minimal" cover.
    
    # Let's do a single pass.
    final_indices = []
    for idx in current_blocks:
        # Check if this block is redundant
        # It is redundant if every t-subset it covers is covered by at least one OTHER block.
        # i.e., count > 1 for all t in block.
        if np.min(cover_counts[block_tsubs[idx]]) > 1:
            # Redundant, remove it
            cover_counts[block_tsubs[idx]] -= 1
            # Do not add to final_indices
        else:
            # Essential, keep it
            final_indices.append(idx)
            
    # Construct the final list of blocks
    result_blocks = [list(all_blocks[i]) for i in final_indices]
    
    return result_blocks
