import itertools
import time
import random
import sys

def _deadline_ok(deadline):
    return time.time() < deadline

def _prune_redundant(selected, block_masks, deadline):
    """Remove redundant blocks from selected set using bitsets."""
    if not selected:
        return selected
    
    # selected is a list of block indices
    # We work on a copy to avoid modification during iteration issues
    sol = list(selected)
    
    changed = True
    while changed and _deadline_ok(deadline):
        changed = False
        for i in range(len(sol)):
            b = sol[i]
            M_b = block_masks[b]
            
            # Compute union of all other blocks
            union_others = 0
            for j in range(len(sol)):
                if j != i:
                    union_others |= block_masks[sol[j]]
            
            # Check if M_b is subset of union_others
            # If M_b & ~union_others is 0, then b is redundant
            if (M_b & ~union_others) == 0:
                sol.pop(i)
                changed = True
                break # Restart the outer loop
    
    return sol

def _local_search(selected, block_masks, subset_to_blocks, n_t, deadline, max_iterations=200):
    """Try to improve the solution by swapping blocks."""
    if not selected or len(selected) < 2:
        return selected
    
    sol = list(selected)
    sol_set = set(sol)
    
    # Precompute total union for quick check? 
    # We'll recompute union_others in each step, which is O(N).
    # N is ~300, so O(N) is fine.
    
    for _ in range(max_iterations):
        if not _deadline_ok(deadline):
            break
        
        if not sol:
            break
            
        # Pick a random block to remove
        idx = random.randint(0, len(sol) - 1)
        b_to_remove = sol[idx]
        M_remove = block_masks[b_to_remove]
        
        # Remove it temporarily
        sol.pop(idx)
        sol_set.remove(b_to_remove)
        
        # Compute union of remaining blocks
        union_remaining = 0
        for b in sol:
            union_remaining |= block_masks[b]
            
        # Uncovered subsets if we keep it removed
        U = M_remove & ~union_remaining
        
        if U == 0:
            # Block was redundant, keep it removed
            continue
        
        # Try to find a replacement block that covers U
        # Pick a random subset from U to find candidates
        # Get index of a set bit in U
        # u_idx = (U & -U).bit_length() - 1  # This gets the lowest set bit
        # But we need the index in the original subset space.
        # U is a bitmask over n_t subsets.
        # lowest_bit = U & -U
        # s_idx = lowest_bit.bit_length() - 1
        
        lowest_bit = U & -U
        s_idx = lowest_bit.bit_length() - 1
        
        candidates = subset_to_blocks[s_idx]
        
        replacement = -1
        for b in candidates:
            if b in sol_set:
                continue
            M_b = block_masks[b]
            # Check if M_b covers all of U
            if (U & ~M_b) == 0:
                replacement = b
                break
                
        if replacement != -1:
            # Accept the swap
            sol.append(replacement)
            sol_set.add(replacement)
        else:
            # Revert
            sol.insert(idx, b_to_remove)
            sol_set.add(b_to_remove)
            
    return sol

def construct(v, k, t):
    start = time.time()
    limit = 120.0
    deadline = start + limit - 5.0  # Safety margin
    
    # 1. Generate all t-subsets and assign indices
    t_subs = list(itertools.combinations(range(v), t))
    n_t = len(t_subs)
    sub_index = {sub: i for i, sub in enumerate(t_subs)}
    
    # 2. Generate all k-blocks and their bitmasks
    blocks = []
    block_masks = []
    
    # Pre-allocate masks? No, just append.
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            idx = sub_index[sub]
            mask |= (1 << idx)
        blocks.append(block)
        block_masks.append(mask)
        
    if not _deadline_ok(deadline):
        return []

    n_blocks = len(blocks)
    
    # Build sub_to_blocks
    sub_to_blocks = [[] for _ in range(n_t)]
    for i, mask in enumerate(block_masks):
        # Iterate through set bits in mask
        m = mask
        while m:
            lb = m & -m
            idx = lb.bit_length() - 1
            sub_to_blocks[idx].append(i)
            m ^= lb
            
    if not _deadline_ok(deadline):
        return []

    # 3. Greedy Set Cover with multiple random restarts
    best_selected = None
    best_size = float('inf')
    
    num_restarts = 15
    top_k = 5
    
    for restart in range(num_restarts):
        if not _deadline_ok(deadline):
            break
        
        # Randomized greedy
        uncovered = (1 << n_t) - 1
        selected = []
        selected_set = set()
        
        while uncovered:
            if not _deadline_ok(deadline):
                break
            
            # Pick a random uncovered subset
            # lowest_bit = uncovered & -uncovered
            # s_idx = lowest_bit.bit_length() - 1
            # To pick random, we can pick a random index from the set bits?
            # Or just pick the lowest one for determinism in restart 0?
            # For randomness, we can shuffle the bits? No, that's expensive.
            # Just pick a random integer and check if bit is set?
            # Or, since n_t is 11628, we can just pick a random index and check.
            # But we need to pick from *uncovered*.
            # Efficient way:
            # 1. Get list of set bits? Expensive.
            # 2. Pick random index 0..n_t-1, if bit not set, retry?
            #    Probability of success is high if few uncovered.
            #    But initially, all are uncovered.
            #    Let's just pick the lowest set bit for simplicity and speed.
            #    Randomization comes from the "top_k" selection.
            
            lowest_bit = uncovered & -uncovered
            s_idx = lowest_bit.bit_length() - 1
            
            candidates = sub_to_blocks[s_idx]
            
            # Calculate gains
            gains = []
            for b in candidates:
                gain = (uncovered & block_masks[b]).bit_count()
                gains.append((gain, b))
            
            # Sort by gain descending
            gains.sort(reverse=True)
            
            # Pick randomly among top candidates
            if restart == 0:
                best_block = gains[0][1]
            else:
                k = min(top_k, len(gains))
                best_block = random.choice(gains[:k])[1]
            
            selected.append(best_block)
            selected_set.add(best_block)
            
            uncovered &= ~block_masks[best_block]
        
        # Prune redundant blocks
        selected = _prune_redundant(selected, block_masks, deadline)
        
        # Local search
        selected = _local_search(selected, block_masks, sub_to_blocks, n_t, deadline, max_iterations=100)
        
        # Update best
        if len(selected) < best_size:
            best_size = len(selected)
            best_selected = list(selected)
        
        # If we have time, do more local search on the best solution
        if _deadline_ok(deadline) and best_selected is not None:
            # Run local search on the current best
            improved_best = _local_search(best_selected, block_masks, sub_to_blocks, n_t, deadline, max_iterations=50)
            if len(improved_best) < best_size:
                best_size = len(improved_best)
                best_selected = improved_best
    
    # Final pruning
    if best_selected is not None:
        best_selected = _prune_redundant(best_selected, block_masks, deadline)
    
    # Convert to result
    result = [list(blocks[i]) for i in best_selected] if best_selected else []
    
    return result
