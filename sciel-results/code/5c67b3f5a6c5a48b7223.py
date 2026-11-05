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
    
    while True:
        if time.time() > deadline - 10.0:
            break
            
        gains = A.dot(active)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        b_idx = np.argmax(gains)
        selected_block_indices.append(b_idx)
        
        covered_indices = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        active[covered_indices] = 0
        
        if np.all(active == 0):
            break
            
    # Local Search: Deletion + Swap
    # Maintain cover counts
    cover_counts = np.zeros(num_t_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
        cover_counts[covered] += 1
    
    # Phase 1: Try to remove redundant blocks
    changed = True
    while changed:
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
    
    # Phase 2: Try swap improvements - replace one block with another
    # that covers more "scarce" t-subsets
    if time.time() < deadline - 5.0:
        selected_set = set(selected_block_indices)
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            # Find the block with the highest "scarcity weight" among its covered t-subsets
            # and try to swap it with a better block
            for i, b_idx in enumerate(selected_block_indices):
                if time.time() > deadline - 5.0:
                    break
                covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                # Compute scarcity score: sum of (1/cover_count) for t-subsets this block covers
                # Higher means this block is more critical
                scarcity = 0.0
                for c in covered:
                    if cover_counts[c] == 1:
                        scarcity += 1000  # Critical: only covered by this block
                    else:
                        scarcity += 1.0 / cover_counts[c]
                
                # Try to find a replacement block that covers more scarce t-subsets
                # and doesn't lose any critical coverage
                best_swap = None
                best_gain = 0
                
                # Sample some candidate blocks (not all, for speed)
                num_candidates = min(200, num_blocks)
                candidates = np.random.choice(num_blocks, size=num_candidates, replace=False)
                
                for cand_idx in candidates:
                    if cand_idx in selected_set:
                        continue
                    
                    cand_covered = A.indices[A.indptr[cand_idx]:A.indptr[cand_idx+1]]
                    
                    # Check: after removing b_idx, are all its t-subsets still covered?
                    # i.e., for each c in covered, is cover_counts[c] > 1 OR c in cand_covered?
                    valid = True
                    new_scarcity = 0.0
                    for c in covered:
                        if cover_counts[c] == 1 and c not in cand_covered:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Compute gain: how many scarce t-subsets does the new block cover
                    # that weren't well-covered before
                    gain = 0
                    for c in cand_covered:
                        if cover_counts[c] == 0:
                            gain += 1000
                        elif cover_counts[c] == 1:
                            gain += 50
                        else:
                            gain += 1 / cover_counts[c]
                    
                    # Penalty: t-subsets that become less covered
                    penalty = 0
                    for c in covered:
                        if c in cand_covered:
                            # Still covered, no penalty
                            pass
                        else:
                            if cover_counts[c] == 1:
                                penalty += 1000
                            elif cover_counts[c] == 2:
                                penalty += 50
                            else:
                                penalty += 1 / cover_counts[c]
                    
                    net_gain = gain - penalty
                    if net_gain > best_gain:
                        best_gain = net_gain
                        best_swap = cand_idx
                
                if best_swap is not None and best_gain > 0:
                    # Perform the swap
                    old_covered = A.indices[A.indptr[b_idx]:A.indptr[b_idx+1]]
                    new_covered = A.indices[A.indptr[best_swap]:A.indptr[best_swap+1]]
                    
                    # Update cover counts
                    for c in old_covered:
                        cover_counts[c] -= 1
                    for c in new_covered:
                        cover_counts[c] += 1
                    
                    selected_block_indices[i] = best_swap
                    selected_set.discard(b_idx)
                    selected_set.add(best_swap)
                    improved = True
                    break  # Restart the scan after a swap
            
    # Convert to output format
    blocks = [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
    return blocks
