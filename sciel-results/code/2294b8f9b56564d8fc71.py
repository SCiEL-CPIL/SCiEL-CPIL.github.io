import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit (or 150s limit per prompt)
    
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
    # Using a sparse-like representation might be better for memory if it was larger, 
    # but for 50k x 3.8k, a dense int8 array is ~200MB, which is okay.
    # However, constructing it efficiently is key.
    
    # We can store for each block, the indices of t-subsets it covers.
    # This is a list of arrays.
    block_cover_indices = []
    for block in k_blocks:
        # Find all t-subsets of this block
        subs = list(itertools.combinations(block, t))
        indices = [t_subset_to_idx[sub] for sub in subs]
        block_cover_indices.append(np.array(indices, dtype=np.int32))
        
    # For the gain calculation, we need fast access.
    # If we use the dense matrix M:
    # M = np.zeros((n_k_blocks, n_t_subsets), dtype=np.int8)
    # for b, inds in enumerate(block_cover_indices):
    #     M[b, inds] = 1
    # This construction might take a few seconds.
    
    # Alternative: Keep block_cover_indices.
    # Gain for block b given uncovered_mask (bool array):
    # gain = uncovered_mask[block_cover_indices[b]].sum()
    # This involves a gather operation.
    # Let's try to optimize the greedy step with this list of indices.
    
    # To make it faster, we can pre-compute a dense matrix if time permits, 
    # or just use the index list.
    # Let's check memory/time. 
    # 50388 blocks. Each has 35 indices.
    # Total indices stored: ~1.7 million ints. Small.
    
    # The bottleneck in the previous code was likely the Python loop over blocks 
    # and inner loop over subsets.
    # With NumPy, `uncovered_mask[indices].sum()` is fast.
    # But doing it for 50k blocks in a Python loop:
    # `gains = np.array([uncovered_mask[inds].sum() for inds in block_cover_indices])`
    # This creates a Python list of 50k numpy scalars. This might be slow (~0.1s per iteration?).
    # If we need 150 iterations, that's 15s.
    
    # Can we vectorize the gain calculation?
    # We can't easily do `M @ uncovered` without constructing M.
    # Let's construct M. It's a one-time cost.
    
    M = np.zeros((n_k_blocks, n_t_subsets), dtype=np.int8)
    for b, inds in enumerate(block_cover_indices):
        M[b, inds] = 1
        
    # Now M is ready.
    # Greedy with randomization
    
    best_blocks = None
    best_len = float('inf')
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        # Randomized Greedy
        uncovered_mask = np.ones(n_t_subsets, dtype=np.int8)
        selected_indices = []
        
        # To add randomization, we can perturb gains slightly
        # or pick randomly among top-k.
        
        while np.any(uncovered_mask):
            # Calculate gains
            gains = M @ uncovered_mask  # Shape (n_k_blocks,)
            
            # Add small noise to break ties or explore
            # noise = rng.random(n_k_blocks) * 0.01
            # gains += noise
            
            # Pick best
            best_block = np.argmax(gains)
            
            if gains[best_block] == 0:
                # Should not happen if cover is possible
                break
                
            selected_indices.append(best_block)
            
            # Update uncovered mask
            # Set to 0 where M[best_block] is 1
            # Using boolean indexing on M row
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0
            
        # End greedy
        
        # Local Search: Redundancy Removal
        # Check if we can remove any block
        if len(selected_indices) > 0:
            # Compute counts for each subset
            # counts = np.sum(M[selected_indices], axis=0)
            # This is efficient.
            sel_matrix = M[selected_indices]
            counts = sel_matrix.sum(axis=0)
            
            redundant_found = True
            while redundant_found and time.time() < deadline:
                redundant_found = False
                new_selected_indices = []
                for i, b_idx in enumerate(selected_indices):
                    # Check if block b_idx is redundant
                    # It is redundant if for all subsets s it covers, counts[s] > 1
                    # Note: counts currently includes b_idx.
                    # So if min(counts[covered_by_b_idx]) > 1, then removing b_idx leaves count >= 1.
                    
                    subs = block_cover_indices[b_idx]
                    if len(subs) > 0:
                        min_count = counts[subs].min()
                    else:
                        min_count = 0 # Should not happen for k>=t
                    
                    if min_count > 1:
                        # Redundant
                        redundant_found = True
                        # Update counts
                        counts[subs] -= 1
                        # Do not add to new list
                    else:
                        # Keep
                        new_selected_indices.append(b_idx)
                        
                if len(new_selected_indices) < len(selected_indices):
                    selected_indices = new_selected_indices
                    # Note: we don't need to recompute counts from scratch immediately 
                    # if we updated them in the loop, but the loop updates counts 
                    # as we remove. However, the check `counts[subs].min()` relies 
                    # on counts being up-to-date for the *remaining* set + current candidate.
                    # The logic above:
                    # 1. Iterate through current selected_indices.
                    # 2. If redundant, decrement counts for its subsets.
                    # 3. If not, keep it.
                    # This works because we are processing them one by one.
                    # However, if A is redundant and we remove it, counts drop.
                    # Then when we check B, counts reflect A's removal.
                    # This is a valid sequential removal.
            
            # End local search
            
            current_len = len(selected_indices)
            if current_len < best_len:
                best_len = current_len
                best_blocks = [list(k_blocks[i]) for i in selected_indices]
        
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
        while np.any(uncovered_mask):
            gains = M @ uncovered_mask
            best_block = np.argmax(gains)
            if gains[best_block] == 0:
                break
            selected_indices.append(best_block)
            M_row = M[best_block]
            uncovered_mask[M_row == 1] = 0
            
        # Redundancy removal for this one too
        if len(selected_indices) > 0:
            sel_matrix = M[selected_indices]
            counts = sel_matrix.sum(axis=0)
            redundant_found = True
            while redundant_found:
                redundant_found = False
                new_selected_indices = []
                for b_idx in selected_indices:
                    subs = block_cover_indices[b_idx]
                    if len(subs) > 0:
                        min_count = counts[subs].min()
                    else:
                        min_count = 0
                    if min_count > 1:
                        redundant_found = True
                        counts[subs] -= 1
                    else:
                        new_selected_indices.append(b_idx)
                selected_indices = new_selected_indices
        best_blocks = [list(k_blocks[i]) for i in selected_indices]
        
    return best_blocks
