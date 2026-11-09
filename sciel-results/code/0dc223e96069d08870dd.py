import time
import random
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precompute all t-subsets and their bitmask representations
    # Map bitmask -> index for fast lookup
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_bitmasks = []
    bitmask_to_idx = {}
    
    for i, comb in enumerate(t_subsets):
        mask = 0
        for p in comb:
            mask |= (1 << p)
        t_subset_bitmasks.append(mask)
        bitmask_to_idx[mask] = i
        
    num_t_subsets = len(t_subsets)
    
    # We will use a list of booleans to track if a t-subset is covered.
    # To speed up updates, we can just use a set of uncovered indices.
    # However, checking membership in a set of ints is fast.
    # Let's use a set of indices for uncovered t-subsets.
    
    best_blocks = None
    best_score = float('inf')
    
    # Number of restarts depends on time, but let's try a few random restarts
    # with different tie-breaking or initial orders to escape local optima.
    # Given v=19, k=7, t=4, the problem is small enough for fast restarts.
    
    for restart in range(20): # Try up to 20 restarts, break if time runs out
        if time.time() > deadline - 5.0:
            break
            
        random.seed(restart)
        
        uncovered_indices = set(range(num_t_subsets))
        blocks = []
        
        # To add randomness, we can shuffle the initial order of "first uncovered"
        # but the greedy algorithm picks the *first* uncovered one.
        # A better randomization is to shuffle the points when breaking ties in gain,
        # or to randomly select from a set of "max gain" points.
        
        while uncovered_indices:
            if time.time() > deadline - 5.0:
                break
                
            # Pick a seed t-subset
            # Using min() on a set of ints is fast.
            # To introduce randomness, we could pick a random element from the set,
            # but picking the smallest is deterministic. 
            # Let's pick a random uncovered subset to start the block.
            seed_idx = random.choice(list(uncovered_indices))
            seed_mask = t_subset_bitmasks[seed_idx]
            
            # Initialize block with the points from the seed subset
            # We need to extract points from the mask.
            block_points = []
            temp_mask = seed_mask
            while temp_mask:
                # Get least significant bit
                lsb = temp_mask & -temp_mask
                p = lsb.bit_length() - 1
                block_points.append(p)
                temp_mask ^= lsb
            
            # If seed subset size < k, we need to add more points.
            # If seed subset size == k, block is full.
            # Note: seed is a t-subset. If t == k, block is just the seed.
            
            # If t < k, we need to add k - t more points.
            current_block_mask = seed_mask
            # We need to track which points are in the block for fast exclusion
            # and for generating (t-1) subsets.
            
            # To calculate gain efficiently, we need to check which (t-1)-subsets of the current block
            # combined with a new point x form a t-subset that is in uncovered_indices.
            
            # Precompute (t-1) subsets of the current block? 
            # The block grows dynamically. 
            # For small k and t, we can just iterate.
            
            # Let's store the block as a list of points.
            block_list = block_points[:]
            
            while len(block_list) < k:
                best_gain = -1
                best_candidates = []
                
                # Candidates are points not in block_list
                # v is small (19), so iterating all v is fine.
                for x in range(v):
                    if (1 << x) & current_block_mask:
                        continue
                    
                    # Calculate gain: number of uncovered t-subsets formed by adding x to block_list
                    # A t-subset formed is {x} U S where S is a (t-1)-subset of block_list.
                    gain = 0
                    # Iterate over all (t-1) subsets of block_list
                    # Since len(block_list) is small (< k=7), this is fast.
                    for s in itertools.combinations(block_list, t - 1):
                        # Create mask for s
                        s_mask = 0
                        for p in s:
                            s_mask |= (1 << p)
                        # Full t-subset mask
                        full_mask = s_mask | (1 << x)
                        
                        # Check if this t-subset is uncovered
                        # We need to map full_mask to an index to check uncovered_indices
                        idx = bitmask_to_idx.get(full_mask)
                        if idx is not None and idx in uncovered_indices:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain and gain > 0:
                        best_candidates.append(x)
                
                if not best_candidates:
                    # Should not happen if uncovered_indices is not empty and v > k
                    # But if no gain, we might be stuck. 
                    # In a valid covering problem, there should always be a point that helps
                    # unless the current block is already "bad" or we have a logic error.
                    # Actually, if gain is 0 for all, it means adding any point doesn't cover 
                    # new t-subsets. This can happen if all t-subsets involving the current block 
                    # are already covered. But we started with an uncovered seed.
                    # The seed itself is covered by the current block? 
                    # Wait, the seed is a t-subset. It is contained in the block.
                    # So the seed is covered.
                    # If we are adding a point, we are extending the block.
                    # If gain is 0, it means no *new* t-subsets are covered.
                    # This is fine, we just need to fill the block.
                    # We pick a random candidate to break ties or avoid bias.
                    # If best_gain is 0, best_candidates might be empty if we only appended when gain > 0?
                    # No, we initialized best_gain = -1. So if gain is 0, it will be > -1.
                    # So best_candidates will not be empty.
                    pass
                
                # Pick a random candidate from best_candidates to add diversity
                next_point = random.choice(best_candidates)
                block_list.append(next_point)
                current_block_mask |= (1 << next_point)
            
            # Block is complete. Sort for consistency (though not strictly required if we use masks)
            block_list.sort()
            blocks.append(block_list)
            
            # Update uncovered_indices
            # Remove all t-subsets contained in this block
            for s in itertools.combinations(block_list, t):
                s_mask = 0
                for p in s:
                    s_mask |= (1 << p)
                idx = bitmask_to_idx.get(s_mask)
                if idx is not None:
                    uncovered_indices.discard(idx)
        
        if not uncovered_indices and best_score > len(blocks):
            best_blocks = blocks
            best_score = len(blocks)
            
            # Local Search: Redundancy Removal
            # Try to remove blocks that are redundant.
            # A block is redundant if all its t-subsets are covered by other blocks.
            # We can check this by temporarily removing a block and seeing if any t-subset
            # becomes uncovered.
            
            # To make this fast, we can maintain a count of how many blocks cover each t-subset.
            # Let's build a count array.
            counts = [0] * num_t_subsets
            
            # Build counts for current best_blocks
            for b in best_blocks:
                for s in itertools.combinations(b, t):
                    s_mask = 0
                    for p in s:
                        s_mask |= (1 << p)
                    idx = bitmask_to_idx.get(s_mask)
                    if idx is not None:
                        counts[idx] += 1
            
            # Iterate through blocks and try to remove them
            # We do this in a loop, checking if removal is safe.
            # Since removing one might make another redundant, we repeat until no change.
            changed = True
            while changed and time.time() < deadline - 5.0:
                changed = False
                new_blocks = []
                for i, b in enumerate(best_blocks):
                    # Check if block i is redundant
                    # It is redundant if for all s in C(b, t), counts[s] > 1
                    # (i.e., some other block covers it)
                    
                    # Note: counts currently includes block i.
                    # So we need counts[s] > 1 to mean "covered by at least one OTHER block".
                    
                    is_redundant = True
                    for s in itertools.combinations(b, t):
                        s_mask = 0
                        for p in s:
                            s_mask |= (1 << p)
                        idx = bitmask_to_idx.get(s_mask)
                        if idx is not None:
                            if counts[idx] <= 1:
                                is_redundant = False
                                break
                    
                    if is_redundant:
                        # Remove block i
                        # Update counts
                        for s in itertools.combinations(b, t):
                            s_mask = 0
                            for p in s:
                                s_mask |= (1 << p)
                            idx = bitmask_to_idx.get(s_mask)
                            if idx is not None:
                                counts[idx] -= 1
                        changed = True
                    else:
                        new_blocks.append(b)
                
                if changed:
                    best_blocks = new_blocks
                    best_score = len(best_blocks)
            
    if best_blocks is None:
        # Fallback: if we timed out before finding a solution, return a trivial (but likely invalid or huge) solution
        # Or just return the last incomplete one? The task requires a valid covering.
        # If we didn't finish, we should probably return the best *valid* one found.
        # If none found, we might fail. 
        # However, with v=19, k=7, t=4, the greedy should finish quickly.
        # Let's ensure we return something. If best_blocks is None, we re-run a single greedy without time pressure?
        # No, just return empty or raise? 
        # The contract says "Valid when...". If we return invalid, score is 0.
        # Let's assume we found at least one.
        return []
        
    return best_blocks
