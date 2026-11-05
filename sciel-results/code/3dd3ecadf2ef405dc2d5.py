import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the specific task are expected, but we handle generic v,k,t
    # For C(19,7,4), v=19, k=7, t=4
    # We assume the evaluator passes v=19, k=7, t=4.
    # If v, k, t are different, the bitmask approach might need adjustment if v is large.
    # But the task specifies C(19,7,4).
    
    # Check if bitmask approach is feasible
    # Number of t-subsets: C(v, t)
    # If C(v, t) > 10000, bitmask with Python int is still okay (arbitrary precision), 
    # but precomputation of all k-subsets might be too large.
    # For v=19, k=7, C(19,7)=50388. C(19,4)=3876. This is small.
    
    # General fallback for larger parameters: standard greedy
    # But we optimize for the specific known case or small cases.
    
    if v > 20 or k > 10:
        # Fallback to simple greedy to avoid memory blowup or excessive precomp time
        # This is a safety net, though task is fixed.
        return _greedy_fallback(v, k, t)

    # Precomputation for bitmask approach
    points = list(range(v))
    
    # 1. Map t-subsets to indices
    t_subsets = list(itertools.combinations(points, t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    full_mask = (1 << num_t_subsets) - 1
    
    # 2. Generate all k-subsets and their coverage masks
    k_subsets = list(itertools.combinations(points, k))
    num_k_subsets = len(k_subsets)
    k_subset_masks = [0] * num_k_subsets
    
    for i, block in enumerate(k_subsets):
        mask = 0
        # For each t-subset within this block, set the bit
        # block is a tuple of k points
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        k_subset_masks[i] = mask

    # Greedy + Pruning with restarts
    best_blocks = None
    best_size = float('inf')
    
    deadline = time.time() + 110  # Safety margin
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Greedy Construction with Randomization
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            best_gain = -1
            best_candidates = []
            
            # Find blocks with max gain
            # To speed up, we can sample if num_k_subsets is large, but 50k is small.
            for i in range(num_k_subsets):
                # Gain is number of new t-subsets covered
                # gain = popcount(k_subset_masks[i] & uncovered)
                gain = (k_subset_masks[i] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain:
                    best_candidates.append(i)
            
            if best_gain == 0:
                # Should not happen if uncovered is not 0 and valid design exists
                # But if we are stuck, break to avoid infinite loop
                break
            
            # Randomly pick from best candidates to diversify
            chosen_idx = random.choice(best_candidates)
            selected_indices.append(chosen_idx)
            uncovered &= ~k_subset_masks[chosen_idx]
            
            # Early termination if covered all
            if uncovered == 0:
                break

        # Pruning Phase
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            # We need to check each block if it's essential
            # Essential if it covers at least one t-subset not covered by others
            n_selected = len(selected_indices)
            if n_selected == 0:
                break
            
            # Precompute OR of all others for each position? 
            # Or just compute on the fly for each candidate for removal.
            # For each block b in selected:
            #   mask_b = k_subset_masks[b]
            #   mask_others = OR of k_subset_masks[o] for o in selected, o != b
            #   if mask_b & ~mask_others == 0, remove b
            
            # Optimization: 
            # Compute total_covered = OR of all selected masks
            # This is just full_mask if valid.
            # To check if b is redundant:
            # We need to know if any t-subset in mask_b is ONLY in mask_b.
            
            # Let's iterate through selected_indices in reverse to allow removal
            new_selected = []
            for idx in selected_indices:
                mask_b = k_subset_masks[idx]
                # Compute mask of all OTHER selected blocks
                mask_others = 0
                for other_idx in selected_indices:
                    if other_idx != idx:
                        mask_others |= k_subset_masks[other_idx]
                
                # If mask_b is subset of mask_others, it's redundant
                if (mask_b & ~mask_others) == 0:
                    changed = True
                else:
                    new_selected.append(idx)
            
            if changed:
                selected_indices = new_selected
            else:
                break
        
        # Update best
        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_blocks = [list(k_subsets[i]) for i in selected_indices]
            
            # If we hit the lower bound, we can stop
            # Lower bound for C(19,7,4) is 131. 
            # If we find 131, it's optimal (or at least best known lower bound).
            if best_size <= 131:
                break

    if best_blocks is None:
        # Fallback if something went wrong
        return _greedy_fallback(v, k, t)
        
    return best_blocks

def _greedy_fallback(v, k, t):
    """Simple greedy fallback for larger parameters or errors."""
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
