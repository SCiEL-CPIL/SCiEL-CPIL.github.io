import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 110s limit, 10s margin
    
    # Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    n_subsets = len(t_subsets)
    
    # Precompute masks for all possible k-blocks? 
    # C(19,7) = 50388. 
    # Each mask is an int. 
    # Storing 50k ints is fine.
    # Precomputing them saves time in the inner loop.
    all_blocks = list(itertools.combinations(range(v), k))
    block_masks = {}
    for b in all_blocks:
        mask = 0
        for sub in itertools.combinations(b, t):
            mask |= 1 << subset_to_idx[sub]
        block_masks[b] = mask
        
    best_blocks = None
    best_size = float('inf')
    
    for attempt in range(100): # Try many restarts
        if time.time() > deadline:
            break
            
        # Randomize the order of t-subsets to break ties and explore different paths
        # We pick the "first" uncovered subset based on this randomized order
        # Or we can just shuffle the t_subsets list used for selection
        
        # To speed up "find first uncovered", we can maintain a list of indices
        # But with bitsets, finding the lowest set bit is fast (bit_length or & -x)
        
        uncovered = (1 << n_subsets) - 1
        blocks = []
        counts = [0] * n_subsets # For pruning
        
        while uncovered:
            # Find the lowest set bit
            # idx = (uncovered & -uncovered).bit_length() - 1
            # This gives the index of the first uncovered subset
            
            # To introduce randomness, we can pick a random set bit instead of the lowest
            # This is a common improvement for greedy set cover: "randomized greedy"
            # Pick a random uncovered subset
            
            # Method to pick random bit:
            # 1. Get popcount
            # 2. Pick random rank
            # 3. Find k-th set bit
            # This is O(n) in worst case if we scan, but we can use a trick or just iterate if n is small.
            # n=3876. Iterating is fast enough?
            # Alternatively, just use the lowest set bit for determinism in single run, 
            # but we have multiple attempts. 
            # Let's stick to lowest set bit for simplicity and speed, 
            # but shuffle the *initial* mapping? No, mapping is fixed.
            # Let's shuffle the *order* in which we consider candidates for tie-breaking.
            
            idx = (uncovered & -uncovered).bit_length() - 1
            first_subset = t_subsets[idx]
            
            # Start block with first_subset
            block = list(first_subset)
            
            # Add k-t more points
            while len(block) < k:
                best_gain = -1
                best_point = -1
                current_block_mask = 0
                # Calculate mask of current partial block? 
                # We need to evaluate gain of adding x.
                # Gain(x) = popcount( mask(block + [x]) & uncovered )
                
                # Optimization: 
                # We can precompute for each point x, the mask of t-subsets containing x?
                # No, the subset must be a subset of the *block*.
                
                # Let's just compute mask for each candidate.
                # There are v - len(block) candidates.
                # v=19, k=7. Max 16 candidates.
                # Computing mask of 7-block takes 35 ops.
                # 16 * 35 = 560 ops per step.
                # 3 steps per block. 1680 ops per block.
                # 150 blocks * 1680 = 252,000 ops. Very fast.
                
                for x in range(v):
                    if x in block:
                        continue
                    temp_block = tuple(sorted(block + [x]))
                    # We need the mask of this temp_block
                    # But temp_block size is < k. 
                    # The mask should be based on the *final* block? 
                    # No, the gain is how many *currently uncovered* t-subsets are covered by the *potential* full block.
                    # But we don't have the full block yet.
                    # Standard greedy: add point that maximizes coverage of *remaining* uncovered sets.
                    # The "coverage" of a partial block is the union of t-subsets within it.
                    # So yes, compute mask of current partial block + x.
                    
                    mask_x = 0
                    # Generate all t-subsets of temp_block
                    # Since temp_block is sorted, we can iterate combinations
                    for sub in itertools.combinations(temp_block, t):
                         mask_x |= 1 << subset_to_idx[sub]
                         
                    gain = (mask_x & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_mask = mask_x # Store to avoid re-computation if we pick this? 
                        # Actually we re-compute for the next iteration anyway.
                        
                if best_point == -1:
                    # Should not happen if uncovered is not empty and v>k
                    break
                block.append(best_point)
                block.sort()
                
            # Finalize block
            final_block = tuple(block)
            blocks.append(final_block)
            
            # Update uncovered and counts
            mask = block_masks[final_block]
            uncovered &= ~mask
            for sub in itertools.combinations(final_block, t):
                counts[subset_to_idx[sub]] += 1
                
        # Pruning Phase
        # Repeat until no block is removed
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                # Check if blocks[i] is redundant
                redundant = True
                for sub in itertools.combinations(blocks[i], t):
                    if counts[subset_to_idx[sub]] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block
                    b = blocks[i]
                    blocks.pop(i)
                    for sub in itertools.combinations(b, t):
                        counts[subset_to_idx[sub]] -= 1
                    changed = True
                    break # Restart loop as indices changed
            
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            
    if best_blocks is None:
        # Fallback: return empty or a trivial valid covering if time ran out before any success?
        # The problem guarantees a solution exists.
        # If we didn't finish even one greedy pass, we might return an invalid result.
        # But 100 restarts of a fast greedy should finish quickly.
        # If time ran out, we should return the best we have.
        # If best_blocks is None, it means we didn't complete even one construction?
        # Unlikely.
        return []
        
    return [list(b) for b in best_blocks]
