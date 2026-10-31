import itertools
import time
import random
import numpy as np

def _remove_redundant(selected_indices, Cover, cover_counts, deadline):
    """Iteratively remove redundant blocks. Returns updated selected_indices and cover_counts."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_selected_indices = []
        for idx in selected_indices:
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_selected_indices.append(idx)
            else:
                cover_counts -= Cover[idx]
                changed = True
        selected_indices = new_selected_indices
    return selected_indices, cover_counts

def _try_block_replacement(selected_indices, Cover, cover_counts, n_blocks, deadline):
    """Try to replace selected blocks with better ones. Returns updated selected_indices and cover_counts."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_selected_indices = []
        for idx in selected_indices:
            # Check if this block is essential (covers any t-subset with count 1)
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                new_selected_indices.append(idx)
                continue
            
            # Block is not essential. Try to find a replacement that allows removing another block.
            # Strategy: Remove this block, then try to see if we can add one block that covers
            # all the t-subsets that were uniquely covered by this block.
            cover_counts -= Cover[idx]
            
            # Find t-subsets that are now uncovered
            uncovered_tsubs = np.where(cover_counts == 0)[0]
            
            if len(uncovered_tsubs) == 0:
                # Block was redundant, keep it removed
                improved = True
                continue
            
            # Try to find a single block that covers all uncovered_tsubs
            # This is expensive, so only try if few tsubs are uncovered
            if len(uncovered_tsubs) <= 10:
                best_replacement = -1
                best_gain = 0
                for candidate in range(n_blocks):
                    if candidate in selected_indices:
                        continue
                    # Check if candidate covers all uncovered_tsubs
                    if np.all(Cover[candidate, uncovered_tsubs] == 1):
                        # Count how many new tsubs this candidate covers
                        gain = np.sum(Cover[candidate] == 1)
                        if gain > best_gain:
                            best_gain = gain
                            best_replacement = candidate
                
                if best_replacement >= 0:
                    # Replace idx with best_replacement
                    cover_counts += Cover[best_replacement]
                    new_selected_indices.append(best_replacement)
                    improved = True
                    continue
            
            # No good replacement found, restore the block
            cover_counts += Cover[idx]
            new_selected_indices.append(idx)
        
        selected_indices = new_selected_indices
    return selected_indices, cover_counts

def _local_optimize(selected_indices, Cover, cover_counts, n_blocks, deadline):
    """Perform local optimization: remove redundant blocks and try replacements."""
    # Phase 1: Remove redundant blocks
    selected_indices, cover_counts = _remove_redundant(selected_indices, Cover, cover_counts, deadline)
    
    # Phase 2: Try block replacements
    selected_indices, cover_counts = _try_block_replacement(selected_indices, Cover, cover_counts, n_blocks, deadline)
    
    # Phase 3: Remove redundant blocks again after replacements
    selected_indices, cover_counts = _remove_redundant(selected_indices, Cover, cover_counts, deadline)
    
    return selected_indices, cover_counts

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Precompute all k-blocks and t-subsets
    points = list(range(v))
    all_kblocks = list(itertools.combinations(points, k))
    all_tsubs = list(itertools.combinations(points, t))
    
    n_blocks = len(all_kblocks)
    n_tsubs = len(all_tsubs)
    
    # Map t-subsets to indices
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # 2. Build coverage matrix
    Cover = np.zeros((n_blocks, n_tsubs), dtype=np.uint8)
    
    for i, B in enumerate(all_kblocks):
        for ts in itertools.combinations(B, t):
            j = tsub_to_idx[ts]
            Cover[i, j] = 1
            
    # 3. Greedy Construction
    selected_indices = []
    uncovered = np.ones(n_tsubs, dtype=np.uint8)
    remaining_time_limit = deadline - 30.0
    
    while np.sum(uncovered) > 0:
        if time.time() > remaining_time_limit:
            break
            
        gains = Cover @ uncovered
        best_idx = np.argmax(gains)
        
        if gains[best_idx] == 0:
            break
            
        selected_indices.append(best_idx)
        uncovered = uncovered & (Cover[best_idx] == 0)
        
    # Convert indices to blocks
    blocks = [all_kblocks[i] for i in selected_indices]
    
    # Post-processing: Local optimization
    if time.time() < deadline - 5.0:
        # Compute initial cover counts
        cover_counts = np.zeros(n_tsubs, dtype=np.int32)
        for idx in selected_indices:
            cover_counts += Cover[idx]
            
        # Run local optimization
        selected_indices, cover_counts = _local_optimize(selected_indices, Cover, cover_counts, n_blocks, deadline - 5.0)
        
        # Update blocks
        blocks = [all_kblocks[i] for i in selected_indices]
        
    return blocks
