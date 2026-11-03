import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    subset_set = set(all_subsets)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts
    while time.time() - start_time < time_limit:
        blocks = greedy_cover(v, k, t, subset_set, random.seed())
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
            # Try to improve by removing redundant blocks
            # We need to be careful with the time here too
            if time.time() - start_time > time_limit - 5:
                break
                
            blocks = remove_redundant_blocks(blocks, v, k, t)
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

    return best_blocks if best_blocks is not None else []

def greedy_cover(v, k, t, subset_set, seed):
    random.seed(seed)
    uncovered = set(subset_set)
    blocks = []
    
    while uncovered:
        # Pick a random uncovered subset to start the block
        # Using random choice from a list is O(N). 
        # Converting set to list every time is expensive.
        # We can just pick the first one, or maintain a list.
        # For simplicity and speed, let's pick a random element.
        # Python's set doesn't support random indexing.
        # We can convert to list, but that's O(N) per block.
        # Number of blocks is ~300. N=11628. 300*11628 ~ 3.5M ops. Acceptable.
        
        # Better: Just pick min(uncovered) for determinism in one run, 
        # but we want randomness. 
        # Let's just pick a random subset from the list of all subsets that is in uncovered.
        # Or simply: pick the first one in a shuffled list? 
        # Actually, the "urgency" heuristic suggests picking the one with fewest covering options.
        # But for now, random choice is a good diversifier.
        
        # To avoid O(N) list conversion every iteration, we can just pick 
        # a random point from the universe? No, we need a subset.
        
        # Let's use a simple heuristic: Pick the uncovered subset with the 
        # "lowest" value in a random permutation?
        # Simpler: Convert to list once per block construction step? 
        # No, uncovered changes.
        
        # Let's just pick min(uncovered) but randomize the "value" by hashing?
        # Or just pick the first element of a randomly shuffled list of uncovered.
        
        # Optimization: 
        # The number of uncovered subsets decreases.
        # Let's just pick a random subset from the full list and check if it's uncovered?
        # No, that's inefficient if coverage is high.
        
        # Let's stick to: 
        # 1. If uncovered is large, pick min.
        # 2. Randomize the order of points considered in the greedy fill.
        
        # Actually, a very common trick is to just pick the first uncovered subset 
        # but randomize the initial "permutation" of points or subsets.
        
        # Let's try: Pick a random subset from `subset_set` (the full list) 
        # and check if it's in `uncovered`. If not, try again.
        # This is O(1) expected if coverage is low, but gets slow near the end.
        
        # Better: Maintain a list of uncovered subsets? 
        # Removing from a list is O(N).
        
        # Let's use the `min` approach but with a randomized tie-breaker or 
        # just rely on the randomization of the *greedy fill* and *restarts*.
        # The seed program used `min(uncovered)`. 
        # Let's use `random.choice(list(uncovered))`.
        
        seed_subset = random.choice(list(uncovered))
        
        block = list(seed_subset)
        random.shuffle(block) # Just to be sure order doesn't bias
        
        while len(block) < k:
            # Find best point to add
            best_gain = -1
            best_point = -1
            candidates = [x for x in range(v) if x not in block]
            random.shuffle(candidates) # Randomize tie-breaking
            
            for x in candidates:
                # Calculate gain: how many uncovered t-subsets are covered by adding x
                # S is covered if x in S and S\{x} in block
                gain = 0
                # Iterate over (t-1) subsets of block
                for rest in itertools.combinations(block, t-1):
                    # Check if rest + (x,) is in uncovered
                    # We need to check if the set (rest U {x}) is in uncovered
                    # rest is a tuple of t-1 elements.
                    # We need to form the tuple of t elements.
                    # Sort rest + (x,) to make a canonical tuple
                    new_sub = tuple(sorted(rest + (x,)))
                    if new_sub in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1:
                # Should not happen if uncovered is not empty and k < v
                break
                
            block.append(best_point)
            
            # Early termination for this block if no more gain?
            # If best_gain == 0, adding more points won't cover new subsets 
            # that contain the current block's (t-1) subsets?
            # Wait, if we add x, we cover subsets containing x and t-1 from block.
            # If we add y next, we cover subsets containing y and t-1 from block U {x}.
            # So gain can be non-zero even if previous gain was 0? 
            # No, if gain(x) was 0, it means no uncovered subset has x and t-1 in block.
            # But adding y might cover subsets with y and t-1 in block U {x}.
            # So we should continue.
            
            block.sort()
            
        # Add block to list
        block = tuple(sorted(block))
        blocks.append(block)
        
        # Update uncovered
        for s in itertools.combinations(block, t):
            uncovered.discard(s)
            
    return blocks

def remove_redundant_blocks(blocks, v, k, t):
    # Count coverage
    cover_count = {}
    for b in blocks:
        for s in itertools.combinations(b, t):
            cover_count[s] = cover_count.get(s, 0) + 1
            
    changed = True
    while changed:
        changed = False
        new_blocks = []
        for b in blocks:
            # Check if b is redundant
            redundant = True
            for s in itertools.combinations(b, t):
                if cover_count.get(s, 0) < 2:
                    redundant = False
                    break
            
            if redundant:
                # Remove b
                for s in itertools.combinations(b, t):
                    cover_count[s] -= 1
                changed = True
            else:
                new_blocks.append(b)
        blocks = new_blocks
        
    return blocks
