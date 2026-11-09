"""Improved program for covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    # Deadline guard: return best result found before 140s (safety margin from 150s limit)
    start_time = time.time()
    deadline = start_time + 140
    
    # Precompute all t-subsets as tuples for fast lookup
    all_t_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_t_subsets)
    blocks = []
    
    # For efficiency, maintain a mapping from points to count of uncovered t-subsets containing them
    # This helps with the greedy selection of points within a block
    
    while uncovered:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Start a new block with the lexicographically smallest uncovered t-subset
        # Actually, let's be smarter: pick an uncovered t-subset that appears in fewest potential blocks
        # But for now, use a greedy heuristic: pick the uncovered t-subset with points that are
        # in the most uncovered t-subsets (to maximize coverage)
        
        # Simple approach: pick the first uncovered t-subset (deterministic)
        # Better: pick a t-subset where the sum of "coverage potential" of its points is high
        # For simplicity and speed, let's just pick min(uncovered)
        
        block = list(min(uncovered))
        
        # Greedily add points to maximize coverage of remaining uncovered t-subsets
        while len(block) < k:
            if time.time() > deadline:
                break
            
            best_point = -1
            best_gain = -1
            block_set = set(block)
            
            # For each candidate point, compute how many new t-subsets it would cover
            # A new t-subset is covered if it contains x and t-1 points from block
            # and the t-subset is currently uncovered
            
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets that contain x and t-1 elements from block
                gain = 0
                for combo in itertools.combinations(block, t - 1):
                    # Create the t-subset: combo + (x,)
                    t_subset = tuple(sorted(combo + (x,)))
                    if t_subset in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                break
            
            block.append(best_point)
        
        if len(block) < k:
            # Couldn't complete the block before deadline, skip
            if time.time() > deadline:
                break
            # Fill with arbitrary remaining points to make it a valid block
            block_set = set(block)
            for x in range(v):
                if len(block) >= k:
                    break
                if x not in block_set:
                    block.append(x)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered set
        for t_sub in itertools.combinations(block, t):
            uncovered.discard(t_sub)
    
    # If we ran out of time but still have uncovered subsets, we need to return something valid
    # The problem requires a valid covering. If we didn't finish, we need to complete it.
    # However, completing it might take too long. Let's check if we have a valid covering.
    
    # Actually, the contract says "Valid when every t-subset lies in at least one block"
    # If we return an incomplete covering, it's invalid and scores 0.
    # So we must ensure validity. Let's do a final check and if needed, add more blocks.
    
    # Check if covering is complete
    if uncovered:
        # We need to add more blocks to cover the remaining t-subsets
        # Use a fast greedy approach for the remainder
        while uncovered:
            if time.time() > deadline - 5:  # Leave 5s safety margin
                # Emergency: add blocks covering single t-subsets (worst case)
                # This will be slow but should finish
                for t_sub in list(uncovered):
                    if time.time() > deadline:
                        break
                    block = list(t_sub)
                    block_set = set(block)
                    # Fill to size k
                    for x in range(v):
                        if len(block) >= k:
                            break
                        if x not in block_set:
                            block.append(x)
                            block_set.add(x)
                    block.sort()
                    blocks.append(block)
                    for ts in itertools.combinations(block, t):
                        uncovered.discard(ts)
                break
            
            # Normal greedy for remaining
            block = list(min(uncovered))
            block_set = set(block)
            while len(block) < k:
                best_point = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for combo in itertools.combinations(block, t - 1):
                        t_subset = tuple(sorted(combo + (x,)))
                        if t_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point == -1:
                    break
                block.append(best_point)
                block_set.add(best_point)
            
            if len(block) < k:
                for x in range(v):
                    if len(block) >= k:
                        break
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
            
            block.sort()
            blocks.append(block)
            for t_sub in itertools.combinations(block, t):
                uncovered.discard(t_sub)
    
    return blocks
# EVOLVE-BLOCK-END
