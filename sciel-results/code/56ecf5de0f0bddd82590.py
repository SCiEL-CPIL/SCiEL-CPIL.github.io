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

def _try_block_swap(selected_indices, Cover, cover_counts, n_blocks, deadline):
    """Try 1-for-1 block swaps to reduce block count or improve coverage."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n_sel = len(selected_indices)
        if n_sel == 0:
            break
        # Shuffle for diversity
        order = list(range(n_sel))
        random.shuffle(order)
        for pos in order:
            if time.time() > deadline:
                break
            idx = selected_indices[pos]
            # Check if essential (covers any t-subset with count exactly 1)
            if np.any((Cover[idx] == 1) & (cover_counts == 1)):
                continue
            # Temporarily remove this block
            cover_counts -= Cover[idx]
            uncovered = np.where(cover_counts == 0)[0]
            if len(uncovered) == 0:
                # Block was redundant
                selected_indices.pop(pos)
                improved = True
                break
            # Try to find a replacement that covers all uncovered tsubs
            if len(uncovered) <= 30:
                best_c = -1
                best_score = -1
                for c in range(n_blocks):
                    if c in selected_indices:
                        continue
                    # Check if c covers all uncovered
                    if np.all(Cover[c, uncovered] == 1):
                        # Score: total coverage of c (prefer denser blocks)
                        score = int(np.sum(Cover[c]))
                        if score > best_score:
                            best_score = score
                            best_c = c
                if best_c >= 0:
                    cover_counts += Cover[best_c]
                    selected_indices[pos] = best_c
                    improved = True
                    break
            # Restore
            cover_counts += Cover[idx]
    return selected_indices, cover_counts

def _local_optimize(selected_indices, Cover, cover_counts, n_blocks, deadline):
    """Perform local optimization: remove redundant blocks and try replacements."""
    # Phase 1: Remove redundant blocks
    selected_indices, cover_counts = _remove_redundant(selected_indices, Cover, cover_counts, deadline)
    
    # Phase 2: Try block swaps (1-for-1)
    selected_indices, cover_counts = _try_block_swap(selected_indices, Cover, cover_counts, n_blocks, deadline)
    
    # Phase 3: Remove redundant blocks again
    selected_indices, cover_counts = _remove_redundant(selected_indices, Cover, cover_counts, deadline)
    
    # Phase 4: More swaps if time allows
    if time.time() < deadline:
        selected_indices, cover_counts = _try_block_swap(selected_indices, Cover, cover_counts, n_blocks, deadline)
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
    remaining_time_limit = deadline - 45.0
    
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
