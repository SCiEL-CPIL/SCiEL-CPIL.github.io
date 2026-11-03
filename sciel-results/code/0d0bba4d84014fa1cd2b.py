import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Pre-compute all triples for faster checking? 
    # Actually, checking `tuple in set` is O(1).
    # We need the set of uncovered triples.
    
    all_points = list(range(v))
    all_triples = set(itertools.combinations(all_points, t))
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple iterations
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        uncovered = set(all_triples)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered triple to start the block
            # This adds randomness to the greedy choice
            start_triple = random.choice(list(uncovered))
            block = list(start_triple)
            
            while len(block) < k:
                best_gain = -1
                best_point = None
                candidates = [x for x in all_points if x not in block]
                
                # Calculate gain for each candidate
                # Gain = number of currently uncovered triples that would be covered
                # by adding this point to the current block.
                # A triple is formed by choosing t-1 points from block and the new point x.
                # We need to check if that triple is in `uncovered`.
                
                # Optimization: Pre-compute the (t-1)-subsets of the current block?
                # They change as we add points.
                # For t=3, t-1=2. We check pairs in block.
                
                for x in candidates:
                    gain = 0
                    # Generate all (t-1) combinations from current block
                    for rest in itertools.combinations(block, t - 1):
                        # Create the triple
                        # We need to sort to match the canonical form in `uncovered`
                        # Since block elements are added in arbitrary order, 
                        # and start_triple is sorted, but x is not necessarily > max(block)
                        # So we must sort.
                        triple = tuple(sorted(rest + (x,)))
                        if triple in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                    elif gain == best_gain and best_point is not None:
                        # Tie-breaking: randomize to allow exploration
                        if random.random() < 0.5:
                            best_point = x
                            
                if best_point is None:
                    # Should not happen if uncovered is not empty and v > k
                    break
                block.append(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            block_triples = set(itertools.combinations(block, t))
            uncovered.difference_update(block_triples)
            
            # Early termination if we already have a worse result than best
            # and no chance of improvement? 
            # Hard to know. But if len(blocks) >= best_count, we might stop this iteration?
            # No, we need to finish the cover to prune.
            # But if len(blocks) is already much larger, we can abort.
            if len(blocks) > best_count + 10: 
                 # Heuristic: if we are already 10 blocks worse, abort this run
                 # This is risky but saves time. Let's be conservative.
                 pass

        # Pruning Phase
        # Count coverage
        from collections import Counter
        cover_count = Counter()
        for b in blocks:
            for tr in itertools.combinations(b, t):
                cover_count[tr] += 1
        
        changed = True
        current_blocks = blocks
        while changed:
            changed = False
            new_blocks = []
            for b in current_blocks:
                # Check if any triple in b is covered exactly once
                essential = False
                for tr in itertools.combinations(b, t):
                    if cover_count[tr] == 1:
                        essential = True
                        break
                if essential:
                    new_blocks.append(b)
                else:
                    # Remove this block
                    changed = True
                    # Decrement counts
                    for tr in itertools.combinations(b, t):
                        cover_count[tr] -= 1
            
            current_blocks = new_blocks
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks
            
        # If we found a very good solution, we might want to restart with different randomness
        # The loop continues naturally.
        
    return best_blocks
