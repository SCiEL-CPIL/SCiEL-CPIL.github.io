import itertools
import random
import time

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy Set Cover followed by Local Search Minimization.
    """
    # 1. Greedy Construction
    # Initialize uncovered subsets
    all_subsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_subsets)
    blocks = []
    
    # Pre-compute all possible k-blocks? No, too many for v=24, k=7.
    # C(24,7) is ~496,512. This is manageable.
    # But iterating through all of them for each step is slow.
    # Standard greedy: pick the block that covers the most uncovered subsets.
    
    # To speed up greedy, we can just generate candidates or use a heuristic.
    # For v=24, k=7, we can pre-compute the subsets for each block?
    # Memory: 496k blocks * 35 subsets each. ~17M integers. This is fine.
    
    # However, a simpler greedy:
    # Start with an empty block, add points one by one maximizing gain.
    # This is the "Greedy Block Construction" used in the seed.
    # It is fast but often leaves redundancy.
    
    # Let's use the seed's greedy construction to get a starting point, 
    # but then apply a strong local search.
    
    # Seed Greedy:
    uncovered = set(all_subsets)
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            best_gain = -1
            best_point = -1
            current_block_set = set(block)
            for x in range(v):
                if x in current_block_set:
                    continue
                # Calculate gain
                gain = 0
                # Subsets of (block + x) of size t that include x
                # are (t-1)-subsets of block combined with x
                for rest in itertools.combinations(block, t - 1):
                    sub = tuple(sorted(list(rest) + [x]))
                    if sub in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            block.append(best_point)
        block.sort()
        blocks.append(block)
        # Update uncovered
        for sub in itertools.combinations(block, t):
            uncovered.discard(sub)
            
    # 2. Local Search Minimization
    # We want to remove blocks.
    # A block can be removed if every t-subset within it is covered by at least one other block.
    
    # Data structures for efficient coverage counting
    # Map from subset (tuple) to count of blocks covering it
    coverage_counts = {}
    for b in blocks:
        for sub in itertools.combinations(b, t):
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
    # Helper to check if a block is redundant
    def is_redundant(block, coverage_counts):
        for sub in itertools.combinations(block, t):
            if coverage_counts[sub] <= 1:
                return False
        return True

    # Helper to remove a block
    def remove_block(block, coverage_counts, blocks):
        for sub in itertools.combinations(block, t):
            coverage_counts[sub] -= 1
        blocks.remove(block)

    # Helper to add a block
    def add_block(block, coverage_counts, blocks):
        for sub in itertools.combinations(block, t):
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
        blocks.append(block)
        
    # Minimization Loop
    start_time = time.time()
    time_limit = 110  # Leave 10s for safety
    
    while time.time() - start_time < time_limit:
        removed_any = False
        
        # Pass 1: Try to remove any redundant block
        # Iterate in reverse to allow safe removal from list
        for i in range(len(blocks) - 1, -1, -1):
            if is_redundant(blocks[i], coverage_counts):
                remove_block(blocks[i], coverage_counts, blocks)
                removed_any = True
                # Restart removal pass to catch newly redundant blocks
                break
                
        if removed_any:
            continue
            
        # Pass 2: If no block can be removed, try to swap points to enable removal
        # Randomly pick a block and try to swap one point
        if not blocks:
            break
            
        # Limit number of swap attempts per second to avoid infinite tight loops
        # or just run until time limit
        block_idx = random.randint(0, len(blocks) - 1)
        block = blocks[block_idx]
        
        # Try swapping a point in this block
        # Pick a random point in block to remove
        x_idx = random.randint(0, k - 1)
        x = block[x_idx]
        
        # Find a candidate y not in block
        candidates = [y for y in range(v) if y not in block]
        if not candidates:
            continue
        y = random.choice(candidates)
        
        # Check if swapping x with y in this block helps
        # We don't want to break coverage.
        # The new block is block' = (block - {x}) U {y}
        # We need to ensure that all subsets in block' are covered.
        # Actually, we just need to ensure the *design* remains valid.
        # Since we are modifying an existing block, we must ensure that any subset 
        # that was ONLY covered by the old block (specifically those involving x) 
        # is still covered.
        
        # Subsets involving x in old block:
        # These are subsets of (block - {x}) of size t-1, combined with x.
        # If any of these has coverage count 1, and the new block doesn't cover it 
        # (which it won't, since x is removed), we break coverage.
        
        # So, we can only swap if for all subsets S in (block - {x}) C (t-1):
        # coverage_counts[S + {x}] > 1
        # OR
        # The new block covers S + {x}? No, new block doesn't have x.
        # So we MUST have coverage_counts[S + {x}] > 1 for all such S.
        
        can_swap = True
        old_block_set = set(block)
        new_block_list = [p for p in block if p != x]
        new_block_list.append(y)
        new_block_set = set(new_block_list)
        
        # Check subsets lost
        for rest in itertools.combinations(block, t - 1):
            if x in rest:
                continue
            # rest does not contain x, so sub = rest + (x,) is a subset of old block
            # Wait, combinations of block of size t-1.
            # If x is not in rest, then rest + (x,) is a t-subset of old block.
            # If x is in rest, then rest is a subset of (block - {x})? No.
            # Let's just iterate all t-subsets of old block that contain x.
            pass
            
        # More efficient check:
        # For each t-subset of old_block that contains x:
        #   if coverage_counts[sub] == 1:
        #       can_swap = False; break
        for sub in itertools.combinations(block, t):
            if x in sub:
                if coverage_counts[sub] == 1:
                    can_swap = False
                    break
        
        if not can_swap:
            continue
            
        # If we can swap, does it help?
        # We accept the swap if it allows the removal of ANY block (including the one we modified, or another one).
        # Let's simulate the swap.
        
        # 1. Remove old block from coverage counts
        for sub in itertools.combinations(block, t):
            coverage_counts[sub] -= 1
            
        # 2. Add new block to coverage counts
        for sub in itertools.combinations(new_block_list, t):
            coverage_counts[sub] = coverage_counts.get(sub, 0) + 1
            
        # 3. Check if any block is now redundant
        found_redundant = -1
        for i in range(len(blocks)):
            # Note: blocks[block_idx] is still the old block in the list, 
            # but logically it's the new block. 
            # We should check all blocks.
            # To be efficient, we can just check the modified block and maybe a random few.
            # But checking all is safer.
            b = blocks[i]
            if i == block_idx:
                b = new_block_list
            if is_redundant(b, coverage_counts):
                found_redundant = i
                break
                
        if found_redundant != -1:
            # Accept swap and remove the redundant block
            blocks[block_idx] = sorted(new_block_list)
            # Remove the redundant block
            # Note: if found_redundant == block_idx, we remove the block we just modified?
            # That means the new block is redundant.
            # If found_redundant != block_idx, we remove a different block.
            
            # We need to update blocks list and coverage counts for the removal.
            # The coverage counts currently reflect: Old Block Removed, New Block Added.
            # We need to remove blocks[found_redundant] from counts.
            
            b_to_remove = blocks[found_redundant]
            if found_redundant == block_idx:
                 b_to_remove = new_block_list
            
            for sub in itertools.combinations(b_to_remove, t):
                coverage_counts[sub] -= 1
                
            del blocks[found_redundant]
            
        else:
            # Swap didn't help, revert
            # 1. Remove new block from counts
            for sub in itertools.combinations(new_block_list, t):
                coverage_counts[sub] -= 1
            # 2. Add old block to counts
            for sub in itertools.combinations(block, t):
                coverage_counts[sub] += 1

    return blocks
