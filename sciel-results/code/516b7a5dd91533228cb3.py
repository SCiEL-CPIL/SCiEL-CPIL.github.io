"""Covering design C(19,8,5): Multi-phase local search with swap optimization."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # 1. Generate initial solution using randomized greedy
    def greedy_randomized():
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            first_t = random.choice(list(uncovered))
            block = list(first_t)
            random.shuffle(block)
            
            while len(block) < k:
                best_point = -1
                best_gain = -1
                # Evaluate all candidate points
                candidates = [x for x in range(v) if x not in block]
                random.shuffle(candidates)
                
                for x in candidates:
                    # Calculate gain: how many uncovered t-subsets are covered by adding x
                    gain = 0
                    # A t-subset is covered if it contains x and (t-1) points from current block
                    # We need to check if the subset is currently uncovered
                    for rest in itertools.combinations(block, t - 1):
                        # Form the t-subset
                        sub = tuple(sorted(rest + (x,)))
                        if sub in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        if best_gain == 0:
                            break # No point helps, break early
                
                if best_point != -1:
                    block.append(best_point)
                else:
                    # Fallback if no gain (shouldn't happen if uncovered is not empty and k>t)
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    # 2. Local Search: Redundancy Removal
    def remove_redundant(blocks, deadline):
        # Count coverage of each t-subset
        counts = Counter()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] += 1
        
        i = 0
        while i < len(blocks) and time.time() < deadline:
            b = blocks[i]
            # Check if removing b leaves any t-subset with count 0
            removable = True
            for sub in itertools.combinations(b, t):
                if counts[sub] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove block i
                for sub in itertools.combinations(b, t):
                    counts[sub] -= 1
                blocks.pop(i)
            else:
                i += 1
                
        return blocks

    # 3. Local Search: 1-for-1 Block Swaps
    def optimize_swaps(blocks, deadline):
        # Precompute coverage counts
        counts = Counter()
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] += 1
        
        # List of all t-subsets for quick reference (optional, for debugging/verification)
        all_subs = set(itertools.combinations(range(v), t))
        
        improved = True
        # Limit iterations to avoid infinite loops or excessive time
        max_iters = 200 
        iter_count = 0
        
        while improved and time.time() < deadline and iter_count < max_iters:
            improved = False
            iter_count += 1
            
            # Shuffle blocks to try different orders
            # But we need to maintain the list structure for replacement
            # We'll iterate through blocks and try to replace each one
            
            # To speed up, we can precompute the set of t-subsets for each block
            block_subs = [set(itertools.combinations(b, t)) for b in blocks]
            
            for i in range(len(blocks)):
                if time.time() > deadline:
                    return blocks, counts
                
                current_block = blocks[i]
                current_subs = block_subs[i]
                
                # Calculate "value" of current block: number of t-subsets it uniquely covers
                # Or more simply, we want to find a new block that covers more "rare" subsets
                # A simple heuristic: find a block that maximizes the number of subsets with count 1
                # that are covered by the new block, minus the number of subsets with count 1
                # that are ONLY covered by the current block.
                
                # Let's try a simpler swap:
                # Try to replace block i with a random block that covers at least as many
                # "critical" (count=1) subsets.
                
                # Generate a candidate block
                # A good candidate might be formed by taking a random t-subset with high count
                # or just a random k-subset.
                
                # Let's try a few random candidates
                found_better = False
                for _ in range(20): # Try 20 random swaps per block per iteration
                    if time.time() > deadline:
                        return blocks, counts
                        
                    new_block = tuple(sorted(random.sample(range(v), k)))
                    new_subs = set(itertools.combinations(new_block, t))
                    
                    # Calculate gain
                    # Gain = (# subsets in new_subs with count 1) - (# subsets in current_subs with count 1 that are NOT in new_subs)
                    # Wait, if we replace, we lose current_subs and gain new_subs.
                    # Net change in "safety" is complex.
                    # Simpler: Does the new block cover any subset that the current block doesn't,
                    # while still covering all subsets that the current block uniquely covers?
                    
                    # Let's check if new_block is a superset of the "unique" parts of current_block
                    # Unique parts: subsets in current_subs where counts[sub] == 1
                    unique_current = {sub for sub in current_subs if counts[sub] == 1}
                    
                    # If new_block doesn't cover all unique_current, it's a bad swap (might break coverage)
                    # Actually, if counts[sub]==1, removing current block makes it 0.
                    # So new_block MUST cover all unique_current to be a valid swap (maintaining coverage).
                    
                    if not unique_current.issubset(new_subs):
                        continue
                        
                    # If it is a valid swap, does it improve the count?
                    # Improvement heuristic: Does it cover more subsets in total?
                    # Or does it cover more subsets with count > 1 (making them more redundant)?
                    # Let's just check if it covers any NEW subsets that were not covered by current_block.
                    # If so, it might allow future removals.
                    
                    # Actually, a better metric:
                    # Let's just accept if it covers all unique_current.
                    # This preserves validity. Over time, randomization might find blocks that
                    # are "better" positioned.
                    
                    # To ensure progress, let's require that the new block covers at least one
                    # subset that the current block does NOT cover.
                    if new_subs.issubset(current_subs):
                        continue
                        
                    # Perform swap
                    # Update counts
                    for sub in current_subs:
                        counts[sub] -= 1
                    for sub in new_subs:
                        counts[sub] += 1
                        
                    blocks[i] = list(new_block)
                    block_subs[i] = new_subs
                    improved = True
                    found_better = True
                    break # Move to next block
                    
                    # If we didn't find a better block for this i, we just continue
                    
        return blocks, counts

    # Main Logic
    best_blocks = None
    best_len = float('inf')
    
    # Run a few randomized greedy restarts to find a good starting point
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        blocks = greedy_randomized()
        
        # Optimize
        blocks = remove_redundant(blocks, deadline)
        blocks, counts = optimize_swaps(blocks, deadline)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        # If we have time, we could try more swaps on the best solution
        # But for now, let's just keep the best found so far
        
    # Final check: if we have time, do more swapping on the best solution
    if best_blocks is not None and time.time() < deadline - 5.0:
        # Re-initialize counts for the best blocks
        counts = Counter()
        for b in best_blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] += 1
        
        # Run more swaps
        best_blocks, _ = optimize_swaps(best_blocks, deadline)
        
        # Final redundancy removal
        best_blocks = remove_redundant(best_blocks, deadline)

    if best_blocks is None:
        # Fallback to simple greedy if something went wrong or no time
        best_blocks = greedy_randomized()
        
    return best_blocks
# EVOLVE-BLOCK-END
