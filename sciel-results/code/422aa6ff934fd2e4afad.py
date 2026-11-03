import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    # Use 140s limit to be safe
    deadline = start_time + 140.0
    
    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_t_subsets = len(t_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute B: (n_blocks, C(k,t))
    # C(8,5) = 56
    k_choose_t = np.math.comb(k, t)
    B = np.empty((n_blocks, k_choose_t), dtype=np.int32)
    
    for i, block in enumerate(all_blocks):
        subs = itertools.combinations(block, t)
        idxs = [t_subset_to_idx[s] for s in subs]
        B[i] = idxs
        
    best_blocks = None
    best_count = float('inf')
    
    # Number of restarts
    # Estimate time per restart
    # Greedy: ~300 steps * 5ms = 1.5s
    # Reduction: ~0.1s
    # Total ~1.6s
    # Budget 140s -> ~80 restarts
    # Let's try 50 restarts to be safe.
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0: # Stop if less than 5s left
            break
            
        counts = np.zeros(n_t_subsets, dtype=np.int32)
        selected_idx = []
        
        # Greedy Phase
        while np.any(counts == 0):
            if time.time() > deadline - 2.0:
                break
                
            is_uncovered = (counts == 0)
            # Calculate gains
            # B is (n_blocks, 56). is_uncovered is (n_t_subsets,)
            # is_uncovered[B] is (n_blocks, 56)
            gains = is_uncovered[B].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            max_indices = np.where(gains == max_gain)[0]
            
            if len(max_indices) > 1:
                idx = np.random.choice(max_indices)
            else:
                idx = max_indices[0]
                
            selected_idx.append(idx)
            counts[B[idx]] += 1
            
        # Check if covering is complete
        if np.any(counts == 0):
            continue
            
        # Reduction Phase
        # We have selected_idx (list of block indices)
        # We want to remove as many as possible.
        
        # Convert to numpy array for vectorized initial check
        if not selected_idx:
            continue
            
        sel_array = np.array(selected_idx)
        
        # We will iteratively try to remove blocks.
        # To be efficient, we can just loop through the list.
        # Since N is ~300, and each check is O(56), it's fast.
        
        # Sort by "redundancy" to remove easy ones first?
        # Let's compute min count for each block.
        # block_subsets = B[sel_array]
        # block_counts = counts[block_subsets]
        # min_counts = np.min(block_counts, axis=1)
        
        # We'll just do a sequential pass.
        # To improve, we can shuffle the order or sort by min_count.
        
        # Let's sort by min_count descending (most redundant first)
        block_subsets = B[sel_array]
        block_counts = counts[block_subsets]
        min_counts = np.min(block_counts, axis=1)
        
        # Indices sorted by min_count descending
        order = np.argsort(-min_counts)
        
        new_selected = []
        # We need to update counts as we remove.
        # So we must check against the *current* counts.
        
        for i in order:
            if time.time() > deadline - 1.0:
                break
                
            # Check if block i is removable with current counts
            subs = B[i]
            if np.all(counts[subs] > 1):
                # Remove it
                counts[subs] -= 1
            else:
                new_selected.append(i)
                
        selected_idx = new_selected
        
        if len(selected_idx) < best_count:
            best_count = len(selected_idx)
            best_blocks = [list(all_blocks[i]) for i in selected_idx]
            
    if best_blocks is None:
        # Fallback: return empty or a trivial invalid? 
        # The task says "Valid when...". If we return invalid, score is 0.
        # We should ensure we return a valid covering if possible.
        # The greedy loop usually completes.
        # If it didn't, we might have a partial covering.
        # We should probably just return the best we have, even if invalid?
        # No, "invalid result... scores 0".
        # So we must return a valid covering.
        # If greedy failed, we can try a slower, more robust method?
        # Or just return the first complete covering found.
        
        # Since we break out of restarts if time runs out, 
        # and we only update best_blocks if covering is complete,
        # best_blocks should be set if any restart succeeded.
        
        # If no restart succeeded (e.g. timeout during greedy), 
        # we might have best_blocks = None.
        # In that case, we should return a known valid covering?
        # Or just return []? No, that's invalid.
        
        # Let's add a fallback: Run a simple greedy without numpy if numpy fails?
        # Or just assume numpy works.
        
        # If best_blocks is still None, return empty list (will score 0)
        # But we can try one more simple greedy in pure python?
        # No, time is up.
        
        return []

    return best_blocks
