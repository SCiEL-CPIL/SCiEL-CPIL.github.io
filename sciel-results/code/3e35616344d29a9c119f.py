"""
Combined approach: NumPy Greedy Initialization + Bitmask Local Search (Swaps/Removals)
Target: C(19, 8, 5)
"""
import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    # 145s deadline to leave 5s margin for return and cleanup
    deadline = start_time + 145.0
    
    # 1. Precomputation
    # All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubs)}
    
    # All k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    n_kblocks = len(all_kblocks)
    
    # Precompute masks for each k-block
    # mask[i] is an integer where bit j is 1 if t-subset j is covered by block i
    block_masks = [0] * n_kblocks
    block_subsets_idx = [[] for _ in range(n_kblocks)] # Store indices for faster iteration if needed
    
    for i, block in enumerate(all_kblocks):
        mask = 0
        idxs = []
        for sub in itertools.combinations(block, t):
            idx = tsub_to_idx[sub]
            mask |= (1 << idx)
            idxs.append(idx)
        block_masks[i] = mask
        block_subsets_idx[i] = idxs

    # Full mask representing all t-subsets
    full_mask = (1 << n_tsubs) - 1

    # 2. Greedy Initialization using NumPy for speed
    # We need to know which t-subsets each block covers as a list of indices for NumPy operations
    # Re-use block_subsets_idx
    
    uncovered_mask_np = np.ones(n_tsubs, dtype=bool)
    remaining = n_tsubs
    selected_indices = []
    
    # Precompute gains for all blocks initially? No, that's heavy.
    # We will compute gains on the fly, but vectorized where possible.
    # Actually, for the greedy phase, we can just loop. 
    # To make it faster, we can pre-compute the list of indices for each block.
    
    # Optimization: Instead of full NumPy overhead for 75k blocks * 300 steps,
    # we can use a heuristic: pick blocks that cover rare subsets first?
    # Let's stick to standard greedy but optimized with precomputed lists.
    
    # To speed up the greedy phase significantly, we can use the bitmask approach 
    # but Python loops are slow. 
    # The current program's NumPy approach was:
    # gains[i] = np.count_nonzero(uncovered_mask_np[cover])
    # This is O(N_blocks * N_cover_per_block).
    
    # Let's try a hybrid: 
    # 1. Compute initial gains using NumPy.
    # 2. Update gains efficiently? 
    # Actually, for C(19,8,5), n_kblocks ~ 75k, n_tsubs ~ 11k.
    # Greedy steps ~ 300.
    # 75k * 300 = 22.5M operations. In Python, this is ~10-20s if simple.
    # NumPy vectorization is much faster.
    
    # Let's use the NumPy greedy from the current program as it's known to be fast.
    # We need the array of indices for each block.
    block_cover_arrays = [np.array(idxs, dtype=np.int32) for idxs in block_subsets_idx]
    
    while remaining > 0:
        if time.time() > deadline - 30.0: # Reserve 30s for local search
            break
            
        # Compute gains
        # This is the bottleneck. 
        # We can vectorize the gain calculation if we represent uncovered as a bitset? 
        # No, NumPy boolean array is easier for count_nonzero.
        
        gains = np.zeros(n_kblocks, dtype=np.int32)
        # This loop is slow in Python. 
        # Alternative: Precompute a matrix? 75k x 11k is too big (800MB).
        # Sparse matrix? 
        # Let's stick to the loop but check time.
        
        # Optimization: Only check blocks that have potential gain?
        # All blocks have potential gain until covered.
        
        # Faster greedy: 
        # Pick a random uncovered subset, then pick a block covering it that maximizes gain?
        # This is "Greedy with Randomization" or "Min-Conflicts" style.
        # But standard greedy is usually better for initialization.
        
        # Let's try to speed up the gain calculation.
        # If we maintain a count of how many times each t-subset is covered by *selected* blocks?
        # No, we need *uncovered* count.
        
        # Let's just run the NumPy loop. It was ~10-15s in previous runs.
        for i in range(n_kblocks):
            # If the block doesn't cover any uncovered subset, gain is 0
            # Quick check: if the block's mask & uncovered_mask is 0?
            # We don't have a numpy mask for uncovered easily without conversion.
            # Let's just do the count.
            gains[i] = np.count_nonzero(uncovered_mask_np[block_cover_arrays[i]])
            
        best_idx = int(np.argmax(gains))
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # No block covers any remaining subset? Should not happen if remaining > 0
            # Unless we ran out of time or logic error.
            break
        
        selected_indices.append(best_idx)
        
        # Update uncovered mask
        uncovered_mask_np[block_cover_arrays[best_idx]] = False
        remaining -= best_gain
        
        # Early exit if time is tight
        if time.time() > deadline - 10.0:
            break

    # 3. Convert to Bitmask state for Local Search
    current_mask = 0
    selected_set = set(selected_indices)
    for idx in selected_indices:
        current_mask |= block_masks[idx]
        
    # Ensure we actually covered everything. If not, fill in with simple greedy on remaining
    if current_mask != full_mask:
        uncovered_mask_np = np.ones(n_tsubs, dtype=bool)
        for idx in selected_indices:
            uncovered_mask_np[block_cover_arrays[idx]] = False
            
        remaining = np.count_nonzero(uncovered_mask_np)
        
        # Simple greedy fill
        while remaining > 0 and time.time() < deadline - 5.0:
            gains = np.zeros(n_kblocks, dtype=np.int32)
            for i in range(n_kblocks):
                if i in selected_set: continue
                gains[i] = np.count_nonzero(uncovered_mask_np[block_cover_arrays[i]])
            
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
                
            selected_indices.append(best_idx)
            selected_set.add(best_idx)
            current_mask |= block_masks[best_idx]
            uncovered_mask_np[block_cover_arrays[best_idx]] = False
            remaining -= gains[best_idx]

    # 4. Local Search: Simulated Annealing / Random Walk
    # State: selected_set (set of indices), current_mask
    # Objective: Minimize len(selected_set) while maintaining current_mask == full_mask
    
    # Precompute list of all blocks for random selection
    all_indices = list(range(n_kblocks))
    
    # Helper: Check if removing a block keeps cover valid
    # valid if (current_mask & ~block_masks[idx]) == current_mask
    # i.e., block_masks[idx] is a subset of (current_mask XOR block_masks[idx])? 
    # No, valid if (current_mask ^ block_masks[idx]) == current_mask is FALSE.
    # Valid if (current_mask | block_masks[idx]) == current_mask is always true.
    # Valid if (current_mask & ~block_masks[idx]) == current_mask.
    # This is equivalent to: (block_masks[idx] & ~current_mask) == 0.
    # Since current_mask is full_mask (if valid), ~current_mask is 0.
    # Wait, if we remove a block, new_mask = current_mask ^ block_masks[idx] (if disjoint? No).
    # new_mask = current_mask & ~block_masks[idx] is WRONG.
    # new_mask = current_mask ^ block_masks[idx] is only true if block is unique? No.
    # new_mask = current_mask - block_masks[idx] (set difference).
    # In bitmasks: new_mask = current_mask & ~block_masks[idx] is incorrect because bits might be shared.
    # Correct check: 
    # Let U = current_mask.
    # Let B = block_masks[idx].
    # If we remove idx, new_union = U \ B? No, other blocks might cover B's bits.
    # We need to know if any bit in B is covered ONLY by idx.
    # This requires knowing the count of coverage for each bit, or checking against the union of OTHER blocks.
    
    # Efficient Removal Check:
    # We maintain current_mask.
    # To check if we can remove idx:
    # We need to verify that every t-subset covered by idx is also covered by some other selected block.
    # This is expensive if done naively.
    
    # Alternative: 
    # Keep a count array `cover_counts` of size n_tsubs.
    # cover_counts[j] = number of selected blocks covering t-subset j.
    # Removal is valid if min(cover_counts[j] for j in block_subsets_idx[idx]) > 1.
    
    # Let's maintain cover_counts.
    cover_counts = np.zeros(n_tsubs, dtype=np.int32)
    for idx in selected_indices:
        cover_counts[block_cover_arrays[idx]] += 1
        
    # Verify initial state
    if np.any(cover_counts == 0):
        # Fallback: something went wrong, just return what we have
        return [list(all_kblocks[i]) for i in selected_indices]

    best_size = len(selected_indices)
    best_selected = set(selected_indices)
    best_counts = cover_counts.copy()
    
    # Local Search Parameters
    # We will perform random moves.
    # Move 1: Remove a random block. If valid (min count in its subsets > 1), accept.
    # Move 2: Swap a random selected block with a random unselected block.
    #          Check if new state is valid.
    #          Validity check for swap:
    #          Remove S, Add U.
    #          New counts = cover_counts - counts(S) + counts(U).
    #          Valid if min(new_counts) >= 1.
    #          This is O(k) per swap check (since each block covers 56 subsets).
    #          56 is small. This is fast.
    
    # We will use a simple random walk with restarts or just a long loop.
    # To escape local minima, we accept swaps even if they don't immediately allow removal,
    # but we prioritize moves that allow removal.
    
    # Actually, a "Swap" that doesn't change size is only useful if it enables a future "Remove".
    # So the loop is:
    # 1. Try to remove. If success, continue.
    # 2. If no removal, try a random swap. If valid, accept.
    
    # To avoid getting stuck, we can also try "Add" a block if we are stuck? No, we want min size.
    
    # Let's run this loop until time is up.
    
    current_selected = list(selected_indices)
    current_set = set(selected_indices)
    
    # Precompute list of selected and unselected for faster random choice?
    # current_set is a set. all_indices is a list.
    
    # Optimization: Pre-compute the list of subset indices for each block as a list for fast iteration
    # block_subsets_idx is already a list of lists of ints.
    
    while time.time() < deadline - 2.0:
        # 1. Try to remove a random block
        if current_selected:
            # Pick a random block from current selection
            idx_to_remove = random.choice(current_selected)
            subsets = block_subsets_idx[idx_to_remove]
            
            # Check if all subsets are covered by at least 1 other block
            # i.e., cover_counts[subset] > 1 for all subset in subsets
            can_remove = True
            for s in subsets:
                if cover_counts[s] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Perform removal
                for s in subsets:
                    cover_counts[s] -= 1
                current_selected.remove(idx_to_remove)
                current_set.discard(idx_to_remove)
                
                # Update best
                if len(current_selected) < best_size:
                    best_size = len(current_selected)
                    best_selected = set(current_selected)
                    best_counts = cover_counts.copy()
                    
                # After a successful removal, we might be able to remove more.
                # Loop back to top to try another removal.
                continue
        
        # 2. Try a swap
        # Pick a random selected block S
        if not current_selected:
            break
            
        idx_s = random.choice(current_selected)
        
        # Pick a random unselected block U
        # We need a block not in current_set
        # Randomly pick from all_indices and check? 
        # Or maintain a list of unselected? 
        # Given 75k blocks, checking `if u not in current_set` is O(1) avg.
        # We might pick the same one many times if the set is small.
        # Better: Pick a random index from 0 to n_kblocks-1.
        
        while True:
            idx_u = random.randint(0, n_kblocks - 1)
            if idx_u not in current_set:
                break
        
        # Check validity of swap: Remove S, Add U
        # New counts = cover_counts - counts(S) + counts(U)
        # We need to ensure new_counts[s] >= 1 for all s.
        
        # To do this efficiently, we can just check the subsets of S that are at risk (count == 1)
        # and see if U covers them.
        
        # Let R = {s in subsets(S) | cover_counts[s] == 1}
        # If R is empty, the swap is always valid (removing S doesn't break coverage).
        # If R is not empty, we need U to cover all elements in R.
        
        risky_subsets = []
        for s in block_subsets_idx[idx_s]:
            if cover_counts[s] == 1:
                risky_subsets.append(s)
                
        if risky_subsets:
            # Check if U covers all risky subsets
            # We can check this by iterating risky_subsets and seeing if they are in U's subsets
            # Or using bitmask: 
            # risky_mask = sum(1 << s for s in risky_subsets)
            # if (block_masks[idx_u] & risky_mask) == risky_mask: valid
            
            risky_mask = 0
            for s in risky_subsets:
                risky_mask |= (1 << s)
                
            if (block_masks[idx_u] & risky_mask) != risky_mask:
                # Swap is invalid
                continue # Try next random pair
                
        # Swap is valid!
        # Update state
        # Remove S
        for s in block_subsets_idx[idx_s]:
            cover_counts[s] -= 1
        current_selected.remove(idx_s)
        current_set.discard(idx_s)
        
        # Add U
        for s in block_subsets_idx[idx_u]:
            cover_counts[s] += 1
        current_selected.append(idx_u)
        current_set.add(idx_u)
        
        # Size is same, so no best update here directly.
        # But this new state might allow a removal.
        
    # Restore best state found
    if len(best_selected) < len(current_selected):
        # We need to reconstruct the block list from best_selected
        final_blocks = [list(all_kblocks[i]) for i in best_selected]
    else:
        final_blocks = [list(all_kblocks[i]) for i in current_selected]
        
    # Final validation and cleanup
    # Ensure distinct blocks
    seen = set()
    unique_blocks = []
    for block in final_blocks:
        tblock = tuple(sorted(block))
        if tblock not in seen:
            seen.add(tblock)
            unique_blocks.append(list(tblock))
            
    unique_blocks.sort()
    
    return unique_blocks

