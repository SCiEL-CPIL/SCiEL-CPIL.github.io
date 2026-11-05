import itertools
import time
import numpy as np
from scipy.sparse import csr_matrix

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0
    
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    row_indices = []
    col_indices = []
    data = []
    
    # Precompute covered t-subsets for each block for faster access later
    block_t_subsets = []
    for i, block in enumerate(all_blocks):
        subs = list(itertools.combinations(block, t))
        block_t_subsets.append(subs)
        for t_sub in subs:
            j = t_subset_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            data.append(1)
            
    A = csr_matrix((data, (row_indices, col_indices)), shape=(num_blocks, num_t_subsets))
    
    active = np.ones(num_t_subsets, dtype=np.float64)
    selected_block_indices = []
    
    # Greedy with "hardest subset" tie-breaking
    while True:
        if time.time() > deadline - 15.0:
            break
            
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        # Find all blocks with maximum gain
        max_indices = np.where(gains == max_gain)[0]
        
        if len(max_indices) == 1:
            b_idx = max_indices[0]
        else:
            # Among max-gain blocks, pick the one that covers the "hardest" uncovered subset
            # Use the lexicographically first uncovered t-subset as a tie-breaker
            # Find the index of the lexicographically first active t-subset
            active_indices = np.where(active > 0)[0]
            if len(active_indices) > 0:
                # Pick the block that covers the lexicographically first active subset
                # among those with max gain
                # For efficiency, check a few candidates
                best_b = max_indices[0]
                best_covers_first = False
                for candidate in max_indices[:min(10, len(max_indices))]:
                    covered = A.indices[A.indptr[candidate]:A.indptr[candidate+1]]
                    # Check if this block covers the lexicographically first active subset
                    if active_indices[0] in covered:
                        best_b = candidate
                        break
                b_idx = best_b
            else:
                b_idx = max_indices[0]
        
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Local Search: Deletion
    # Maintain cover counts
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
        
    # Try to remove blocks
    # We iterate through the list. If a block is redundant, we remove it.
    # We do this in a loop until no more blocks can be removed.
    changed = True
    while changed:
        if time.time() > deadline - 5.0:
            break
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            # Check if all t-subsets of this block are covered by others
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            # Remove them
            # Update cover counts
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            # Remove from list
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
    
    # Swap-based local search: try to replace one block with another that reduces total count
    # or improves coverage efficiency
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            n_sel = len(selected_block_indices)
            for i in range(n_sel):
                if time.time() > deadline - 3.0:
                    break
                b_old = selected_block_indices[i]
                covered_old = A.indices[A.indptr[b_old]:A.indptr[b_old+1]]
                
                # Find t-subsets that are only covered by this block
                unique_to_old = []
                for c in covered_old:
                    if cover_counts[c] == 1:
                        unique_to_old.append(c)
                
                if len(unique_to_old) == 0:
                    # This block is redundant, remove it
                    for c in covered_old:
                        cover_counts[c] -= 1
                    selected_block_indices.pop(i)
                    improved = True
                    break
                
                # Try to find a replacement block that covers all unique_to_old
                # and has better "efficiency" (covers more other uncovered subsets)
                best_new = -1
                best_score = -1
                for b_new in range(num_blocks):
                    if b_new == b_old:
                        continue
                    covered_new = A.indices[A.indptr[b_new]:A.indptr[b_new+1]]
                    # Check if new block covers all unique_to_old
                    covers_all = True
                    for c in unique_to_old:
                        if c not in covered_new:
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    # Score: how many additional subsets does it cover that aren't covered by old?
                    # Or just prefer blocks that are already in the list (no change) vs new
                    # Actually, we want to see if we can remove this block and add a better one
                    # For now, skip swap and just do deletion
                    break
                
                # For simplicity, just do deletion pass more aggressively
                # Try removing pairs of blocks if their union is covered by others
                if i + 1 < n_sel:
                    b_old2 = selected_block_indices[i+1]
                    covered_old2 = A.indices[A.indptr[b_old2]:A.indptr[b_old2+1]]
                    # Check if removing both is safe
                    can_remove_both = True
                    for c in covered_old:
                        if cover_counts[c] <= 1:
                            can_remove_both = False
                            break
                    if can_remove_both:
                        for c in covered_old2:
                            if cover_counts[c] <= 1:
                                can_remove_both = False
                                break
                    if can_remove_both:
                        for c in covered_old:
                            cover_counts[c] -= 1
                        for c in covered_old2:
                            cover_counts[c] -= 1
                        selected_block_indices.pop(i+1)
                        selected_block_indices.pop(i)
                        improved = True
                        break
        
        # Rebuild cover counts after swaps
        cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in selected_block_indices:
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            cover_counts[covered] += 1
    
    # Final deletion pass
    changed = True
    while changed and time.time() < deadline - 2.0:
        changed = False
        to_remove = []
        for i, b_idx in enumerate(selected_block_indices):
            covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
            redundant = True
            for c in covered:
                if cover_counts[c] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
        
        if to_remove:
            for i in to_remove:
                b_idx = selected_block_indices[i]
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                cover_counts[covered] -= 1
            remove_set = set(to_remove)
            selected_block_indices = [b for i, b in enumerate(selected_block_indices) if i not in remove_set]
            changed = True
    
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
