import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 110.0  # Leave 10s buffer for safety
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    all_t_subsets_set = set(all_t_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Pre-compute combinations of sizes for efficiency?
    # For a given block B, we need combinations of B of size t.
    
    def remove_redundant(blocks):
        """Remove blocks that are redundant (i.e., all their t-subsets are covered by other blocks)."""
        if not blocks:
            return blocks
        
        # Count coverage for each t-subset
        # Using a dictionary or array might be faster than a set of sets
        # Since v is small (19), we can map tuples to indices or just use a dict
        coverage_count = {s: 0 for s in all_t_subsets_set}
        
        for block in blocks:
            for sub in itertools.combinations(block, t):
                coverage_count[sub] += 1
        
        new_blocks = []
        changed = True
        while changed:
            changed = False
            for block in blocks:
                # Check if this block is redundant
                # A block is redundant if for all sub in C(block, t), coverage_count[sub] > 1
                # (i.e., other blocks still cover it)
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    if coverage_count[sub] <= 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove this block
                    for sub in itertools.combinations(block, t):
                        coverage_count[sub] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            blocks = new_blocks
            new_blocks = []
            # Reset new_blocks for next iteration if changed
            if changed:
                 # We need to re-evaluate because removing one block might make another redundant
                 # But we already appended non-redundant ones to new_blocks in the pass above?
                 # No, the logic above is slightly flawed. 
                 # If I remove a block, the coverage counts change. 
                 # The blocks I kept in new_blocks might now be redundant.
                 # So I should just set blocks = new_blocks and loop again.
                 pass
        
        return blocks

    def greedy_cover(random_seed):
        random.seed(random_seed)
        uncovered = set(all_t_subsets_set)
        blocks = []
        
        # Shuffle the list of uncovered subsets to randomize start
        # But we need to pick from the current uncovered set dynamically
        # So we can't just shuffle the initial list once.
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            start_sub = random.choice(list(uncovered))
            block = list(start_sub)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                best_x = None
                max_gain = -1
                best_candidates = []
                
                # To speed up, we can pre-compute the "potential" subsets
                # But for now, direct calculation is fine for small v
                for x in candidates:
                    new_block = block + [x]
                    # Count how many uncovered t-subsets are covered by new_block
                    # A t-subset is covered if it is a subset of new_block
                    # We only care about those currently in `uncovered`
                    gain = 0
                    # Iterate through all t-subsets of new_block
                    for sub in itertools.combinations(new_block, t):
                        if sub in uncovered:
                            gain += 1
                    
                    if gain > max_gain:
                        max_gain = gain
                        best_candidates = [x]
                    elif gain == max_gain:
                        best_candidates.append(x)
                
                if max_gain == 0:
                    # This shouldn't happen if uncovered is not empty, 
                    # because start_sub is in uncovered and is a subset of block,
                    # so adding any point shouldn't reduce gain below 0?
                    # Actually, if block already covers start_sub, gain for any x is at least 0.
                    # But we need to cover *other* subsets.
                    # If max_gain is 0, we pick a random candidate to avoid infinite loop or bad choices?
                    # It means no single point addition covers any *new* uncovered subset.
                    # This can happen if the uncovered subsets are "far" from the current block.
                    # But we started with an uncovered subset, so the block contains it.
                    # Wait, if block contains start_sub, then start_sub is already "covered" by the block conceptually, 
                    # but we haven't added the block to `blocks` yet, so it's still in `uncovered`.
                    # When we add a point, we check if sub in uncovered.
                    # start_sub is in uncovered. Is start_sub a subset of new_block? Yes.
                    # So gain should be at least 1.
                    # Unless... I am checking `if sub in uncovered`.
                    # Yes, start_sub is in uncovered.
                    # So max_gain >= 1.
                    pass
                
                block.append(random.choice(best_candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                if sub in uncovered:
                    uncovered.discard(sub)
        
        # Post-process: remove redundant blocks
        blocks = remove_redundant(blocks)
        return blocks

    # Run multiple iterations
    seed = 42
    while time.time() < deadline:
        blocks = greedy_cover(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # If we found a very good solution, we might want to try local search further?
            # For now, just keep the best.
        seed += 1
        
        # If we hit the known upper bound, we can try to be more aggressive or just keep searching
        # The known upper bound is 151.
        if best_count <= 151:
             # We are at or below the known best. Keep trying to improve.
             pass

    if best_blocks is None:
        # Fallback: run one deterministic greedy just in case
        return greedy_cover(0)
        
    return best_blocks
