# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Deadline guard: 90s limit, use 85s for safety
    start_time = time.time()
    max_time = 85.0
    
    # Precompute masks as numpy array for vectorization
    all_masks = np.arange(total_masks, dtype=np.int64)
    
    # Helper to check comparability:
    # A is comparable to B if A is subset of B OR B is subset of A
    # (A & B) == A  => A subset B
    # (A & B) == B  => B subset A
    # Since we want strict subset for the pair count, and sets are distinct, 
    # comparability in the lattice sense (A <= B or B <= A) is equivalent 
    # to strict subset for distinct sets.
    
    # We will maintain a list of current masks (Python list for easy manipulation)
    # and a numpy array of the same for vectorized scoring.
    
    def get_initial_selection():
        """
        Greedily construct a good initial set of masks.
        Strategy: Pick masks that have high potential to be comparable to many others.
        A simple heuristic: Start with a long chain, then add sets that are comparable to the most existing sets.
        """
        selected = []
        selected_set = set()
        
        # Start with a random long chain to ensure a strong base of comparable pairs
        perm = list(range(n))
        random.shuffle(perm)
        cur = 0
        chain = [0]
        for p in perm:
            cur |= (1 << p)
            chain.append(cur)
            
        # Add chain elements first
        for mask in chain:
            if mask not in selected_set:
                selected.append(mask)
                selected_set.add(mask)
        
        # Fill the rest greedily
        while len(selected) < m:
            best_mask = -1
            best_score = -1
            
            # Sample a subset of candidates to speed up initialization, 
            # or iterate all if fast enough. 4096 is small.
            # For initialization, let's just pick the mask that maximizes 
            # comparability with the current 'selected' list.
            
            # To make this faster, we can just pick random candidates and keep the best of a batch,
            # or do a full scan if 'selected' is small.
            # Let's do a full scan of all 4096 masks, it's O(4096 * current_size) which is fine.
            
            candidates = [m for m in range(total_masks) if m not in selected_set]
            
            # For efficiency in initialization, we might not need the absolute best, 
            # but let's try to find a good one.
            # Actually, a simpler deterministic init: 
            # Just take the first 40 masks in a specific order that clusters them?
            # No, let's stick to the greedy fill.
            
            # Optimization: Instead of checking all 4096, check a random sample of 100 and pick the best.
            # This is faster and good enough for init.
            sample_size = min(100, len(candidates))
            sampled = random.sample(candidates, sample_size)
            
            for mask in sampled:
                score = 0
                for s in selected:
                    if (mask & s) == mask or (mask & s) == s:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_mask = mask
            
            if best_mask != -1:
                selected.append(best_mask)
                selected_set.add(best_mask)
            else:
                # Fallback: just add a random one
                mask = random.choice(candidates)
                selected.append(mask)
                selected_set.add(mask)
                
        return selected

    def count_pairs_np(masks_np):
        """
        Vectorized count of comparable pairs.
        masks_np is (m, 1) or (m,). Let's use (m, 1) to broadcast with (1, m).
        """
        # masks_np shape (m, 1)
        # other shape (1, m)
        # intersection = masks_np & other  -> (m, m)
        # subset1 = (intersection == masks_np) -> (m, m)
        # subset2 = (intersection == other) -> (m, m)
        # comparable = subset1 | subset2
        # We want upper triangle count excluding diagonal (distinct sets).
        
        inter = masks_np & masks_np.T
        sub1 = (inter == masks_np)
        sub2 = (inter == masks_np.T)
        comp = sub1 | sub2
        
        # Count upper triangle
        # np.triu_indices
        indices = np.triu_indices(m, k=1)
        return int(np.sum(comp[indices]))

    def optimize(masks_list):
        """
        Greedy refinement using NumPy vectorization for candidate evaluation.
        """
        current_masks = list(masks_list)
        current_score = count_pairs_np(np.array(current_masks).reshape(-1, 1))
        
        # Precompute comparability of all 4096 masks against current set?
        # This changes as we swap. So we recompute per pass or per swap.
        # Better: For each position i, we want to find the best replacement.
        # The score change is:
        # Delta = (comparability(new, others) - comparability(old, others))
        # Note: comparability is symmetric.
        
        # We can precompute a "compatibility matrix" C where C[i, j] is 1 if masks[i] and masks[j] are comparable.
        # Then score = sum(C) / 2 (since symmetric) or sum upper triangle.
        # If we replace masks[i] with new_mask:
        # We remove row i and col i from C.
        # We add a new row/col for new_mask.
        # Delta = (sum of new row/col) - (sum of old row/col)
        # Note: The diagonal is 0 (self not comparable in strict sense, or we exclude it).
        # Actually, for distinct sets, self is not comparable.
        
        # Let's use the property:
        # Score = 0.5 * sum_{i != j} C[i, j]
        # If we swap i:
        # New Score = Old Score - sum_{k != i} C[i, k] + sum_{k != i} C_new[i, k]
        # Where C_new[i, k] is 1 if new_mask is comparable to masks[k].
        
        # So for each i, we need:
        # 1. current_degree[i] = sum_{k != i} C[i, k]
        # 2. For each candidate new_mask not in current_masks:
        #    new_degree = sum_{k != i} (1 if new_mask comparable to masks[k])
        #    gain = new_degree - current_degree[i]
        
        # We can vectorize the calculation of "new_degree" for ALL candidates at once for a fixed i?
        # Or for ALL i at once?
        
        # Let's iterate passes. In each pass, iterate i from 0 to m-1.
        # For a fixed i, we have `others` = current_masks without i.
        # We want to find the best `new_mask` from the 4096 available (excluding current ones) 
        # that maximizes comparability with `others`.
        
        # Vectorized approach for fixed i:
        # others_np: shape (m-1, 1)
        # candidates_np: shape (4096, 1)
        # inter = candidates_np & others_np.T  -> (4096, m-1)
        # sub1 = (inter == candidates_np) -> (4096, m-1)
        # sub2 = (inter == others_np.T) -> (4496, m-1) -- wait, broadcasting
        # sub2 = (inter == others_np.T) is correct if others_np.T is (1, m-1) and inter is (4096, m-1)?
        # No, inter is (4096, m-1). others_np.T is (m-1, 1).
        # inter == others_np.T broadcasts to (4096, m-1). Correct.
        # comp = sub1 | sub2 -> (4096, m-1)
        # degrees = np.sum(comp, axis=1) -> (4096,)
        
        # This is 4096 * 39 operations. ~160k ops per i.
        # 40 i's -> 6.4M ops per pass.
        # NumPy handles this very fast.
        
        max_passes = 200 # Can do many more passes now
        improved = True
        
        while improved and time.time() - start_time < max_time - 5.0:
            improved = False
            # Shuffle order to avoid bias
            order = list(range(m))
            random.shuffle(order)
            
            for i in order:
                old_mask = current_masks[i]
                others = [current_masks[k] for k in range(m) if k != i]
                
                if not others:
                    continue
                    
                others_np = np.array(others, dtype=np.int64).reshape(-1, 1)
                
                # Get current degree of old_mask against others
                # We can compute this from the previous matrix or just recompute quickly
                # It's sum of comparability(old_mask, others)
                # Since we are in a loop, let's just use the vectorized candidate check 
                # and find the max. We also need to exclude masks already in current_masks.
                
                # Create a boolean mask of which candidates are valid (not in current_masks)
                # current_masks is a list of ints.
                # We can create a set for O(1) lookup, but we need a vectorized mask for the 4096 array.
                # valid_mask = np.ones(4096, dtype=bool)
                # for cm in current_masks: valid_mask[cm] = False
                
                # This loop is 40 iterations, very fast.
                valid_mask = np.ones(total_masks, dtype=bool)
                for cm in current_masks:
                    valid_mask[cm] = False
                
                # Exclude old_mask? It's already in current_masks, so valid_mask[old_mask] is False.
                # We want to replace old_mask, so we can pick any valid mask.
                
                # Compute degrees for all valid candidates
                # candidates_np = all_masks.reshape(-1, 1)
                # inter = candidates_np & others_np.T
                # sub1 = (inter == candidates_np)
                # sub2 = (inter == others_np.T)
                # comp = sub1 | sub2
                # degrees = np.sum(comp, axis=1)
                
                # To save memory/time, we can do this in chunks or just let NumPy handle it.
                # 4096 x 39 is small.
                
                candidates_np = all_masks.reshape(-1, 1)
                inter = candidates_np & others_np.T
                sub1 = (inter == candidates_np)
                sub2 = (inter == others_np.T)
                comp = sub1 | sub2
                degrees = np.sum(comp, axis=1)
                
                # Mask out invalid candidates
                degrees[~valid_mask] = -1
                
                best_idx = np.argmax(degrees)
                best_gain = degrees[best_idx]
                
                # Calculate current degree of old_mask to determine if we should swap
                # We didn't compute it explicitly in this iteration, but we can infer:
                # If best_gain > current_degree(old_mask), we swap.
                # How to get current_degree(old_mask)?
                # It is the number of comparable pairs involving old_mask.
                # We can compute it quickly:
                old_degree = 0
                for o in others:
                    if (old_mask & o) == old_mask or (old_mask & o) == o:
                        old_degree += 1
                
                if best_gain > old_degree:
                    new_mask = int(all_masks[best_idx])
                    # Perform swap
                    current_masks[i] = new_mask
                    # Update score
                    # Delta = best_gain - old_degree
                    current_score += (best_gain - old_degree)
                    improved = True
        
        return current_masks, current_score

    # Main execution
    best_masks = None
    best_score = -1
    
    # Run a few restarts with different initializations
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > max_time - 10.0:
            break
            
        init_masks = get_initial_selection()
        
        # Optimization
        opt_masks, opt_score = optimize(init_masks)
        
        if opt_score > best_score:
            best_score = opt_score
            best_masks = opt_masks
            
        # Check time
        if time.time() - start_time > max_time - 2.0:
            break

    if best_masks is None:
        # Fallback
        best_masks = [list(range(i)) for i in range(40)] # Invalid distinctness? No, these are distinct.
        # Actually, just return something valid
        best_masks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        
    # Convert best_masks to list of lists
    result = []
    for mask in best_masks:
        s = []
        for b in range(n):
            if mask & (1 << b):
                s.append(b)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
