import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 145s limit, 5s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute the coverage mask for each block
    # Each mask is an integer where bit i is 1 if the block covers t_subsets[i]
    block_masks = [0] * n_blocks
    for i, b in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(b, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Initial state: all subsets are uncovered
    # We use a large integer as a bitset. 
    # n_subsets is 3876, so we need ~60 64-bit words. Python ints handle this natively.
    all_covered_mask = (1 << n_subsets) - 1
    
    best_blocks = None
    best_size = float('inf')
    
    # 2. Greedy Construction with Randomized Restarts
    # We run multiple attempts, each starting with a fresh uncovered mask.
    # Randomization is introduced in tie-breaking to explore different solutions.
    
    # To speed up the search for the best block, we can pre-sort or use heuristics,
    # but for v=19, n_blocks=77520 is small enough that a linear scan is feasible 
    # if we can make the gain calculation fast.
    
    # Gain calculation: 
    # We want the block that maximizes popcount(uncovered & block_mask)
    # Python's int.bit_count() is very fast (C implementation).
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        uncovered = all_covered_mask
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_gain = -1
            best_block_idx = -1
            best_blocks_candidates = []
            
            # To speed up, we can sample a subset of blocks if n_blocks is large, 
            # but here 77k is manageable. However, scanning 77k blocks * 150 times * 100 restarts 
            # might be too slow. 
            # Let's estimate: 77k * 150 (blocks in solution) = 11.5M operations per restart.
            # 11.5M * 100 restarts = 1.15B operations. This is likely too slow in Python.
            
            # Optimization: We should not scan ALL blocks every step.
            # We can maintain a list of candidate blocks or use a heuristic.
            # A common heuristic: pick a random uncovered subset, and only consider blocks containing it.
            
            # Pick a random uncovered subset
            # Get index of a random set bit
            # Method: convert to binary string? No, too slow.
            # Method: use random rank.
            
            # Let's use a different strategy: 
            # 1. Pick a random uncovered subset S.
            # 2. Find all blocks containing S.
            # 3. Among those, pick the one that covers the most OTHER uncovered subsets.
            
            # Precompute: for each t-subset, which blocks contain it?
            # This is a huge memory footprint if stored as lists.
            # Instead, we can just iterate through blocks and check if they contain the subset.
            # But that's still 77k checks.
            
            # Better: Precompute for each t-subset, the list of block indices that contain it.
            # n_subsets = 3876. Each t-subset is contained in C(v-t, k-t) = C(15, 3) = 455 blocks.
            # Total entries: 3876 * 455 ~ 1.7M. This is small and fast to access.
            
            # We haven't precomputed this yet. Let's do it now if we haven't.
            # Actually, doing this inside the loop is bad. Let's precompute it once.
            
            # Wait, I can't easily restructure the whole precomp if I'm inside the loop.
            # Let's precompute subset_to_blocks before the main loop.
            
            break # Break out to restructure precomputation
            
        # Re-precomputation step (move to top)
        
    # Re-do precomputation properly
    # Precompute: for each t-subset index, list of block indices that contain it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for b_idx, b in enumerate(all_blocks):
        for sub in itertools.combinations(b, t):
            s_idx = subset_to_idx[sub]
            subset_to_blocks[s_idx].append(b_idx)
    
    # Now the main loop
    best_blocks = None
    best_size = float('inf')
    
    while time.time() < deadline:
        uncovered = all_covered_mask
        blocks = []
        
        while uncovered:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered subset
            # Find a random set bit in 'uncovered'
            # We can do this by getting the bit length and picking a random position? 
            # No, that's not uniform.
            # We can iterate through bits? Slow.
            # Let's convert to list of set bits? Slow if many bits.
            
            # Efficient way to pick a random set bit:
            # 1. Get the number of set bits (popcount).
            # 2. Pick a random rank r in [0, popcount).
            # 3. Find the r-th set bit.
            
            pop = uncovered.bit_count()
            if pop == 0:
                break
                
            rank = random.randrange(pop)
            
            # Find rank-th set bit
            # We can use a loop, but it might be slow if rank is large.
            # Alternative: just pick the lowest set bit? 
            # The lowest set bit is fast: (uncovered & -uncovered).bit_length() - 1
            # But it's deterministic. To add randomness, we can shuffle the order of checking?
            # Or, we can pick a random block from the candidates of the lowest uncovered subset?
            
            # Let's stick to picking a random uncovered subset using a faster method if possible,
            # or just use the lowest one and rely on random tie-breaking in block selection.
            
            # Let's use the lowest uncovered subset for speed, and randomize the block choice.
            # This is a common greedy heuristic (First-Fit or similar).
            idx = (uncovered & -uncovered).bit_length() - 1
            
            # Get candidate blocks containing this subset
            candidates = subset_to_blocks[idx]
            
            # If no candidates (shouldn't happen), break
            if not candidates:
                break
                
            # Among candidates, find the one that covers the most uncovered subsets
            best_gain = -1
            best_block_idx = -1
            best_gain_blocks = []
            
            for b_idx in candidates:
                mask = block_masks[b_idx]
                gain = (uncovered & mask).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_gain_blocks = [b_idx]
                elif gain == best_gain:
                    best_gain_blocks.append(b_idx)
                    
            if best_gain_blocks:
                # Randomly pick from the best blocks to introduce variation
                selected_block_idx = random.choice(best_gain_blocks)
                blocks.append(selected_block_idx)
                uncovered &= ~block_masks[selected_block_idx]
                
        # End while uncovered
        
        if not blocks:
            continue
            
        # Check if we covered everything
        if uncovered != 0:
            # Incomplete covering, skip pruning? Or try to repair?
            # Greedy usually completes. If not, it's not a valid solution.
            # We can try to add more blocks, but we are already in the loop.
            # If we broke out due to deadline, it might be incomplete.
            # We only consider complete coverings.
            continue
            
        # Pruning Phase
        # Remove redundant blocks
        # A block is redundant if every t-subset it covers is covered by at least one other block in the set.
        
        # Count how many times each t-subset is covered
        # We can use a dictionary or array of counts.
        # n_subsets is 3876, so an array of size 3876 is fine.
        counts = [0] * n_subsets
        
        for b_idx in blocks:
            # Iterate through t-subsets in this block
            b = all_blocks[b_idx]
            for sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[sub]
                counts[s_idx] += 1
                
        # Identify redundant blocks
        to_remove = []
        for i, b_idx in enumerate(blocks):
            b = all_blocks[b_idx]
            redundant = True
            for sub in itertools.combinations(b, t):
                s_idx = subset_to_idx[sub]
                if counts[s_idx] == 1:
                    redundant = False
                    break
            if redundant:
                to_remove.append(i)
                
        # Remove redundant blocks
        if to_remove:
            # Create a set of indices to remove for O(1) lookup
            remove_set = set(to_remove)
            new_blocks = []
            for i, b_idx in enumerate(blocks):
                if i not in remove_set:
                    new_blocks.append(b_idx)
            
            # Update counts for the removed blocks? 
            # Actually, we just need to keep the new list. 
            # But we need to ensure the remaining blocks still form a covering.
            # Since we only removed blocks where all subsets had count > 1, 
            # the remaining blocks still cover all subsets.
            
            blocks = new_blocks
            
            # We can run pruning again? Sometimes removing one block makes another redundant.
            # Let's do a second pass if time permits.
            if time.time() < deadline - 5:
                 counts = [0] * n_subsets
                 for b_idx in blocks:
                     b = all_blocks[b_idx]
                     for sub in itertools.combinations(b, t):
                         s_idx = subset_to_idx[sub]
                         counts[s_idx] += 1
                         
                 to_remove = []
                 for i, b_idx in enumerate(blocks):
                     b = all_blocks[b_idx]
                     redundant = True
                     for sub in itertools.combinations(b, t):
                         s_idx = subset_to_idx[sub]
                         if counts[s_idx] == 1:
                             redundant = False
                             break
                     if redundant:
                         to_remove.append(i)
                         
                 if to_remove:
                     remove_set = set(to_remove)
                     blocks = [b_idx for i, b_idx in enumerate(blocks) if i not in remove_set]
                     
        # Update best
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks[:]
            
    # End while time
    
    if best_blocks is None:
        # Fallback: return empty or a simple greedy result if we failed
        # This should not happen if we at least run one full greedy
        return []
        
    # Convert indices back to blocks
    result_blocks = [list(all_blocks[idx]) for idx in best_blocks]
    return result_blocks
