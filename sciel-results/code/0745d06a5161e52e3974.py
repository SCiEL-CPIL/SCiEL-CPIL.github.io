import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 105.0  # Safety margin
    
    # Precomputation
    points = list(range(v))
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(points, t))
    n_t_subsets = len(t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(points, k))
    n_k_blocks = len(k_blocks)
    
    # Build the incidence matrix M (n_k_blocks x n_t_subsets)
    # M[b, s] = 1 if t-subset s is contained in k-block b
    M = np.zeros((n_k_blocks, n_t_subsets), dtype=np.int8)
    block_cover_indices = []
    for b, block in enumerate(k_blocks):
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[sub] for sub in subs]
        block_cover_indices.append(np.array(indices, dtype=np.int32))
        M[b, indices] = 1
        
    # Greedy with randomization and local search
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Precompute random permutations for tie-breaking
    # We will use a noise vector to perturb gains
    
    while time.time() < deadline:
        # Randomized Greedy
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        used_blocks = np.zeros(n_k_blocks, dtype=bool)
        
        while np.any(uncovered_mask):
            # Calculate gains
            # gains = M @ uncovered_mask
            # To speed up, we can try to only consider blocks that have high potential.
            # But M @ uncovered_mask is a matrix-vector product (50k x 3.8k).
            # This is ~190M operations. With int8, it might be fast.
            
            gains = M @ uncovered_mask
            
            # Mask out used blocks
            gains[used_blocks] = 0
            
            # Add small noise to break ties and explore
            # We want to pick from top candidates sometimes
            # If we just add noise, we might pick a slightly worse block, leading to diversity.
            noise = rng.random(n_k_blocks) * 0.1
            gains_noisy = gains + noise
            
            # Pick best
            best_block = np.argmax(gains_noisy)
            
            if gains[best_block] == 0:
                # No block covers any new subset?
                # This means we have covered everything or we are stuck.
                # Since it's a covering problem, if uncovered_mask is not 0,
                # there must be a block covering it.
                # Unless we used all blocks? No, we have 50k blocks.
                # So this should not happen unless uncovered_mask is 0.
                break
                
            selected_indices.append(best_block)
            used_blocks[best_block] = True
            
            # Update uncovered mask
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0
            
        # End greedy
        
        # Local Search: Redundancy Removal
        if len(selected_indices) > 0:
            # Compute counts for each subset
            # counts = np.sum(M[selected_indices], axis=0)
            sel_matrix = M[selected_indices]
            counts = sel_matrix.sum(axis=0)
            
            # Iteratively remove redundant blocks
            # A block is redundant if all subsets it covers are covered by other blocks
            # i.e., counts[s] > 1 for all s in block.
            
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_selected_indices = []
                # We need to be careful with counts.
                # If we remove a block, counts decrease.
                # We can iterate through the list and check.
                
                # To make this faster, we can pre-compute the subsets for each block.
                # block_cover_indices is already there.
                
                # However, checking min(counts[subs]) > 1 is O(35) per block.
                # 150 blocks * 35 = 5250 ops. Very fast.
                
                # But we need to update counts if we remove a block.
                # If we remove block i, counts[subs_i] -= 1.
                # This affects subsequent checks.
                
                # Let's do a single pass first, then repeat if changes occurred.
                # This is a greedy local search.
                
                for i, b_idx in enumerate(selected_indices):
                    subs = block_cover_indices[b_idx]
                    if len(subs) > 0:
                        # Check if all subsets are covered by others
                        # This is equivalent to: for all s in subs, counts[s] > 1
                        # Note: counts currently includes b_idx.
                        # So if min(counts[subs]) > 1, then removing b_idx leaves count >= 1.
                        if np.min(counts[subs]) > 1:
                            # Redundant
                            changed = True
                            counts[subs] -= 1
                            # Do not add to new list
                        else:
                            # Keep
                            new_selected_indices.append(b_idx)
                    else:
                        # Should not happen for k>=t
                        new_selected_indices.append(b_idx)
                        
                selected_indices = new_selected_indices
                
            # End local search loop
            
            current_len = len(selected_indices)
            if current_len < best_len:
                best_len = current_len
                best_blocks = [list(k_blocks[i]) for i in selected_indices]
                
                # If we hit the lower bound or a very good number, we might want to stop early?
                # Lower bound is 131.
                if best_len <= 131:
                    break

    # End time loop
    
    if best_blocks is None:
        # Fallback: return a trivial valid cover? 
        # Or just the last one?
        # If time was 0, we might have failed.
        # A greedy without restarts is safer.
        # But the loop above runs at least once if time permits.
        # If time was 0, we might have best_blocks None.
        # Let's do a single deterministic greedy if best_blocks is None.
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        used_blocks = np.zeros(n_k_blocks, dtype=bool)
        
        while np.any(uncovered_mask):
            gains = M @ uncovered_mask
            gains[used_blocks] = 0
            best_block = np.argmax(gains)
            
            if gains[best_block] == 0:
                break
                
            selected_indices.append(best_block)
            used_blocks[best_block] = True
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0
            
        # Redundancy removal for this one too
        if len(selected_indices) > 0:
            sel_matrix = M[selected_indices]
            counts = sel_matrix.sum(axis=0)
            changed = True
            while changed:
                changed = False
                new_selected_indices = []
                for b_idx in selected_indices:
                    subs = block_cover_indices[b_idx]
                    if len(subs) > 0:
                        if np.min(counts[subs]) > 1:
                            changed = True
                            counts[subs] -= 1
                        else:
                            new_selected_indices.append(b_idx)
                    else:
                        new_selected_indices.append(b_idx)
                selected_indices = new_selected_indices
                
        best_blocks = [list(k_blocks[i]) for i in selected_indices]
        
    return best_blocks
