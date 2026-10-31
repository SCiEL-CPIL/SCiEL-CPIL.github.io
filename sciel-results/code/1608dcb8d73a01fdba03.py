import itertools
import random
import time

def construct(v, k, t):
    # Parameters for the specific task, but function should be general
    # v=19, k=7, t=4 expected
    
    # 1. Greedy Initialization
    def greedy_cover():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick a random uncovered subset to start the block to diversify
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Find the point that covers the most uncovered subsets
                best_point = None
                best_gain = -1
                # Optimization: Precompute (t-1) subsets of current block
                current_t_minus_1 = list(itertools.combinations(block, t - 1))
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    for rest in current_t_minus_1:
                        new_subset = tuple(sorted(rest + (x,)))
                        if new_subset in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                if best_point is None:
                    # Fallback: pick any random point not in block
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        best_point = random.choice(candidates)
                    else:
                        break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        return blocks

    # 2. Fast Local Search Optimizer
    def optimize(blocks, time_limit=110):
        # State: list of blocks, coverage_count dict
        current_blocks = [tuple(b) for b in blocks]
        
        def calc_coverage(blist):
            cov = {}
            for b in blist:
                for sub in itertools.combinations(b, t):
                    cov[sub] = cov.get(sub, 0) + 1
            return cov

        coverage_count = calc_coverage(current_blocks)
        
        # Helper to check if a block is redundant (all its t-subsets have count > 1)
        def is_redundant(b, cov):
            for sub in itertools.combinations(b, t):
                if cov[sub] <= 1:
                    return False
            return True

        start_time = time.time()
        no_improve_streak = 0
        
        while time.time() - start_time < time_limit:
            improved = False
            
            # Phase 1: Remove redundant blocks
            i = 0
            while i < len(current_blocks):
                b = current_blocks[i]
                if is_redundant(b, coverage_count):
                    # Remove b
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                    current_blocks.pop(i)
                    improved = True
                    # Do not increment i, check the next block at the same index
                else:
                    i += 1
            
            if improved:
                no_improve_streak = 0
                continue # Restart loop as indices changed
            else:
                no_improve_streak += 1
                # If stuck, try swaps or break if time is low
                if no_improve_streak > 100 and time.time() - start_time > time_limit - 10:
                    break
                if time.time() - start_time > time_limit - 5:
                    break

            # Phase 2: Try to swap blocks to enable further removals or reduce size
            if len(current_blocks) > 1:
                # Pick a random index to swap
                idx = random.randint(0, len(current_blocks) - 1)
                old_block = current_blocks[idx]
                
                # Generate a random new block
                new_block = tuple(sorted(random.sample(range(v), k)))
                
                # Check if swapping old_block with new_block is valid
                # We need to ensure that all t-subsets of old_block that are NOT covered by new_block 
                # are still covered by other blocks.
                
                # Efficient check:
                # 1. Identify t-subsets of old_block that have count == 1 (unique to old_block).
                #    If any of these are NOT in new_block, the swap is invalid.
                
                valid_swap = True
                for sub in itertools.combinations(old_block, t):
                    if coverage_count[sub] == 1:
                        # This subset is only covered by old_block
                        if sub not in new_block:
                            valid_swap = False
                            break
                            
                if valid_swap:
                    # Perform swap
                    # Remove old_block
                    for sub in itertools.combinations(old_block, t):
                        coverage_count[sub] -= 1
                    # Add new_block
                    for sub in itertools.combinations(new_block, t):
                        coverage_count[sub] = coverage_count.get(sub, 0) + 1
                    current_blocks[idx] = new_block
                    improved = True
                    no_improve_streak = 0
                    # Go back to Phase 1 to see if this enabled removals
                    continue

        return [list(b) for b in current_blocks]

    # 3. Cleanup: Remove redundant blocks (just in case)
    def clean_up(blocks):
        block_set = [tuple(b) for b in blocks]
        coverage_count = {}
        for b in block_set:
            for sub in itertools.combinations(b, t):
                coverage_count[sub] = coverage_count.get(sub, 0) + 1
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in block_set:
                redundant = True
                for sub in itertools.combinations(b, t):
                    if coverage_count[sub] <= 1:
                        redundant = False
                        break
                if redundant:
                    changed = True
                    for sub in itertools.combinations(b, t):
                        coverage_count[sub] -= 1
                else:
                    new_blocks.append(b)
            block_set = new_blocks
        
        return [list(b) for b in block_set]

    # Main Execution
    # Run greedy multiple times and pick the best
    best_initial = None
    for _ in range(20):
        initial_blocks = greedy_cover()
        if best_initial is None or len(initial_blocks) < len(best_initial):
            best_initial = initial_blocks
            
    # Run Optimizer
    optimized_blocks = optimize(best_initial, time_limit=110)
    
    # Final cleanup
    final_blocks = clean_up(optimized_blocks)
    
    return final_blocks
