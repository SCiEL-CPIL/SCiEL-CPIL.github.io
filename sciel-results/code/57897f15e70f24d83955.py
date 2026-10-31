import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    num_blocks = len(all_blocks)
    num_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices for fast lookup
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Build coverage matrix
    # masks[i, j] is True if block i covers t-subset j
    masks = np.zeros((num_blocks, num_tsubs), dtype=bool)
    
    # Pre-compute t-subsets for each block to speed up mask filling?
    # Or just fill directly.
    for i, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            masks[i, tsub_to_idx[ts]] = True
            
    # Greedy Max Coverage
    uncovered = np.ones(num_tsubs, dtype=bool)
    selected_indices = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Calculate gain for each block
        # gains[i] = number of uncovered t-subsets covered by block i
        gains = np.sum(masks & uncovered, axis=1)
        
        # Select block with maximum gain
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # No block can cover any remaining t-subset?
            # This implies no valid covering exists or we are stuck.
            # But for C(19,7,4) a valid covering exists.
            break
            
        selected_indices.append(best_idx)
        
        # Update uncovered set
        uncovered = uncovered & ~masks[best_idx]
        
    # Redundancy Removal
    # We want to minimize the number of blocks.
    # A block is redundant if every t-subset it covers is also covered by at least one other selected block.
    
    if not selected_indices:
        return []
        
    current_indices = list(selected_indices)
    
    # Loop until no more blocks can be removed
    changed = True
    while changed and time.time() < deadline:
        changed = False
        
        # Calculate coverage counts for current set
        counts = np.zeros(num_tsubs, dtype=int)
        for idx in current_indices:
            counts += masks[idx]
            
        new_current_indices = []
        for idx in current_indices:
            # Check if block idx is redundant
            # It is redundant if min(counts[ts] for ts in block idx) > 1
            # i.e., every t-subset in this block is covered by at least 2 blocks (this one + at least one other)
            
            # Get counts for t-subsets in this block
            # masks[idx] is a boolean mask of shape (num_tsubs,)
            # counts[masks[idx]] gives the counts for the relevant t-subsets
            
            # If the block covers no t-subsets (impossible for k>=t), skip check
            if np.any(masks[idx]):
                if np.min(counts[masks[idx]]) > 1:
                    # Redundant
                    changed = True
                else:
                    new_current_indices.append(idx)
            else:
                # Block covers nothing? Should not happen.
                new_current_indices.append(idx)
                
        current_indices = new_current_indices
        
    return [list(all_blocks[idx]) for idx in current_indices]
