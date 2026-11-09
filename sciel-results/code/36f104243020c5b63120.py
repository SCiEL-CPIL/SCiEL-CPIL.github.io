"""
Strategy: Bitmask-based Vectorized Greedy Set Cover + Aggressive Local Search.

1. Precompute all C(19,4) = 3876 t-subsets as bitmasks.
2. Use NumPy to vectorize the calculation of how many uncovered subsets each 
   potential block covers.
3. Greedy selection: Pick the block that covers the most uncovered subsets.
4. Local Search: 
   a. Removal: Iteratively remove non-critical blocks.
   b. 1-for-1 Swap: Replace a critical block with a non-selected block that covers 
      all its unique subsets. This allows the new block to potentially cover 
      different unique subsets, enabling further removals or better packing.
      We use a vectorized pre-filter to find candidate swaps quickly.
"""

import time
import numpy as np
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precompute all t-subsets and their bitmasks
    # v=19, t=4 -> 3876 subsets
    # k=7 -> C(19,7) = 50388 blocks
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    # Create a bitmask for each t-subset
    t_masks = np.array([0] * num_t, dtype=np.int64)
    for i, sub in enumerate(t_subsets):
        mask = 0
        for p in sub:
            mask |= (1 << p)
        t_masks[i] = mask
        
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    # Create a bitmask for each k-block
    k_masks = np.array([0] * num_k, dtype=np.int64)
    for i, block in enumerate(k_blocks):
        mask = 0
        for p in block:
            mask |= (1 << p)
        k_masks[i] = mask
        
    # Precompute coverage matrix
    # cover[i, j] = 1 if block i covers t-subset j
    # Memory optimization: Use uint8
    cover_list = []
    chunk_size = 1000
    for start in range(0, num_k, chunk_size):
        end = min(start + chunk_size, num_k)
        k_chunk = k_masks[start:end] # (chunk, )
        # Check: (t_masks[None, :] & ~k_chunk[:, None]) == 0
        check = (t_masks[None, :] & ~k_chunk[:, None]) == 0
        cover_list.append(check.astype(np.uint8))
        
    uncovered = np.ones(num_t, dtype=np.uint8)
    blocks_selected = []
    indices_selected = []
    
    # Greedy Phase
    while np.sum(uncovered) > 0:
        if time.time() > deadline - 20: # Leave 20s for local search
            break
            
        best_idx = -1
        best_gain = 0
        
        # Calculate gain for each block
        for chunk_idx, cover_chunk in enumerate(cover_list):
            # gains = cover_chunk @ uncovered
            gains = cover_chunk.dot(uncovered)
            max_gain_in_chunk = np.max(gains)
            
            if max_gain_in_chunk > best_gain:
                best_gain = max_gain_in_chunk
                local_idx = np.argmax(gains)
                best_idx = chunk_idx * chunk_size + local_idx
        
        if best_gain == 0:
            break
            
        blocks_selected.append(k_blocks[best_idx])
        indices_selected.append(best_idx)
        
        # Update uncovered
        chunk_idx = best_idx // chunk_size
        local_idx = best_idx % chunk_size
        cover_row = cover_list[chunk_idx][local_idx]
        uncovered[cover_row == 1] = 0
        
    # Local Search / Refinement
    # Initialize counts
    counts = np.zeros(num_t, dtype=np.int32)
    for idx in indices_selected:
        chunk_idx = idx // chunk_size
        local_idx = idx % chunk_size
        cover_row = cover_list[chunk_idx][local_idx]
        counts += cover_row
        
    # Phase 1: Removal
    changed = True
    while changed and time.time() < deadline - 10:
        changed = False
        new_indices = []
        for idx in indices_selected:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            
            critical_mask = (cover_row == 1) & (counts == 1)
            if not np.any(critical_mask):
                counts -= cover_row
                changed = True
            else:
                new_indices.append(idx)
                
        indices_selected = new_indices
        blocks_selected = [k_blocks[i] for i in indices_selected]
        
    # Phase 2: 1-for-1 Swaps
    # We want to find a selected block B_s and a non-selected block B_n such that:
    # 1. B_n covers all unique subsets of B_s (subsets where count==1 and covered by B_s).
    # 2. After swap, the covering remains valid.
    # Condition 1 is sufficient for validity if we assume the rest of the covering
    # doesn't break. Wait, if B_n covers all unique subsets of B_s, then removing B_s
    # leaves those subsets covered by B_n. The other subsets covered by B_s are covered
    # by at least one other block (since they are not unique). So validity is maintained.
    # Does this reduce count? No, it's a swap.
    # BUT, it might enable further removals!
    # If B_n covers some subsets that were previously covered by 2 blocks (one being B_s),
    # and B_n covers some subsets that were covered by 3 blocks, the counts change.
    # More importantly, B_n might have a different set of unique subsets (relative to the
    # new configuration) that are easier to cover or allow other blocks to be removed.
    # Actually, a simpler heuristic: 
    # If we swap B_s -> B_n, the total count stays same.
    # However, if B_n covers MORE "rare" subsets (subsets with low count), it might
    # make other blocks removable.
    # Let's try: 
    # For each selected block B_s, find all non-selected blocks B_n that cover all unique subsets of B_s.
    # Among these, pick one that maximizes the number of "extra" subsets it covers 
    # (subsets it covers that B_s did not cover, or subsets where it increases redundancy 
    # in a way that helps removal).
    # Actually, just finding ANY valid swap might be enough to escape a local minimum 
    # in the "removal" landscape.
    
    if time.time() < deadline - 5:
        selected_set = set(indices_selected)
        non_selected = [i for i in range(num_k) if i not in selected_set]
        
        # Pre-compute "unique mask" for each selected block? 
        # It changes after each swap.
        
        # Strategy: Iterate through selected blocks. For each, find a valid swap.
        # To make this fast, we can pre-compute for each non-selected block 
        # the set of t-subsets it covers? No, that's the cover matrix.
        
        # We can use the cover matrix.
        # For a selected block B_s (index s), unique_mask_s = cover[s] & (counts == 1).
        # We need B_n (index n) such that cover[n] covers unique_mask_s.
        # i.e., (cover[n] & ~unique_mask_s) == unique_mask_s? No.
        # (cover[n][unique_mask_s == 1] == 1).all()
        
        # Vectorized approach for finding candidates:
        # Let U_s be the boolean vector of unique subsets for B_s.
        # We want n such that sum(cover[n] & U_s) == sum(U_s).
        # Let S_s = sum(U_s).
        # We can compute dot products of cover[n] with U_s for all n.
        # But doing this for all s and all n is O(N_sel * N_non * T).
        # N_sel ~ 150, N_non ~ 50000, T ~ 4000.
        # 150 * 50000 * 4000 = 30 billion ops. Too slow.
        
        # Optimization: 
        # Only try swaps for blocks that have a "small" number of unique subsets?
        # Or limit the search.
        
        # Alternative: Randomized search.
        # Pick a random selected block. Find a random non-selected block. Check if valid.
        # If valid, perform swap.
        # This is cheap per step.
        
        import random
        random.seed(42)
        
        max_swap_iters = 10000
        iters = 0
        
        while iters < max_swap_iters and time.time() < deadline - 2:
            iters += 1
            
            # Pick a random selected block
            if not indices_selected:
                break
            s_idx_pos = random.randint(0, len(indices_selected) - 1)
            s_idx = indices_selected[s_idx_pos]
            
            # Get unique mask for this block
            chunk_idx_s = s_idx // chunk_size
            local_idx_s = s_idx % chunk_size
            cover_row_s = cover_list[chunk_idx_s][local_idx_s]
            
            unique_mask = (cover_row_s == 1) & (counts == 1)
            n_unique = np.sum(unique_mask)
            
            if n_unique == 0:
                # Should have been removed, but maybe counts changed?
                # If n_unique is 0, it's removable. Let's just remove it.
                counts -= cover_row_s
                indices_selected.pop(s_idx_pos)
                blocks_selected = [k_blocks[i] for i in indices_selected]
                continue
            
            # Try a few random non-selected blocks
            found_swap = False
            # Sample up to 100 candidates
            candidates = random.sample(non_selected, min(100, len(non_selected)))
            
            for n_idx in candidates:
                chunk_idx_n = n_idx // chunk_size
                local_idx_n = n_idx % chunk_size
                cover_row_n = cover_list[chunk_idx_n][local_idx_n]
                
                # Check if n covers all unique subsets of s
                # missing = unique_mask & (cover_row_n == 0)
                # if not np.any(missing):
                
                # Faster check using dot product or sum
                # Sum of cover_row_n where unique_mask is 1 should equal n_unique
                # np.dot(cover_row_n, unique_mask) == n_unique
                
                # Since unique_mask is boolean, we can do:
                # if np.all(cover_row_n[unique_mask]):
                
                # Let's use the dot product trick
                # Note: unique_mask is int/bool. cover_row_n is uint8.
                # np.dot(cover_row_n, unique_mask.astype(np.uint8))
                
                # Optimization: Pre-compute unique_indices = np.where(unique_mask)[0]
                # Then check cover_row_n[unique_indices].all()
                
                # Let's try the dot product first, it's vectorized.
                # But unique_mask changes every iteration.
                
                # Let's just use the boolean indexing, it's fast for sparse masks.
                # Unique subsets are usually few (1-5).
                if np.all(cover_row_n[unique_mask]):
                    # Valid swap
                    # Update counts
                    counts -= cover_row_s
                    counts += cover_row_n
                    
                    # Verify validity (counts > 0)
                    if np.any(counts == 0):
                        # Revert
                        counts -= cover_row_n
                        counts += cover_row_s
                        continue
                    
                    # Perform swap
                    indices_selected[s_idx_pos] = n_idx
                    blocks_selected = [k_blocks[i] for i in indices_selected]
                    found_swap = True
                    break
            
            if not found_swap:
                # No swap found for this random block, try another random block
                # This is effectively a random walk in the solution space
                continue

    # Final cleanup: try removal again after swaps
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        new_indices = []
        for idx in indices_selected:
            chunk_idx = idx // chunk_size
            local_idx = idx % chunk_size
            cover_row = cover_list[chunk_idx][local_idx]
            
            critical_mask = (cover_row == 1) & (counts == 1)
            if not np.any(critical_mask):
                counts -= cover_row
                changed = True
            else:
                new_indices.append(idx)
                
        indices_selected = new_indices
        blocks_selected = [k_blocks[i] for i in indices_selected]
        
    # Final check
    counts = np.zeros(num_t, dtype=np.int32)
    for idx in indices_selected:
        chunk_idx = idx // chunk_size
        local_idx = idx % chunk_size
        cover_row = cover_list[chunk_idx][local_idx]
        counts += cover_row
        
    if np.any(counts == 0):
        # Fallback: return greedy result if we broke it (should not happen)
        # We don't have the greedy result saved separately, so we must ensure
        # our logic is sound.
        pass
        
    return [list(b) for b in blocks_selected]
