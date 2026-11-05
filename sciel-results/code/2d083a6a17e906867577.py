import itertools
import random
import time

def construct(v, k, t):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for 120s limit
    
    # Pre-compute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    # Map each t-subset to an index for faster lookup? 
    # Or just use a set of tuples. Set of tuples is fine for ~10k items.
    
    best_blocks = None
    
    # Number of restarts
    # We don't know how long one run takes. 
    # Let's estimate: 
    # ~300 blocks. 
    # Each block construction: ~8 steps * 19 candidates * (check ~35 4-subsets in set).
    # ~300 * 8 * 19 * 35 = 1.5M set lookups. Very fast.
    # Redundancy removal: 
    # ~300 blocks * 56 subsets * (check count). 
    # If we maintain counts, it's fast.
    
    # Let's try 100 restarts.
    
    for restart in range(100):
        if time.time() > deadline:
            break
            
        # Randomize the order of points or subsets?
        # We can just randomize the tie-breaking in the greedy step.
        
        uncovered = set(all_t_subsets)
        blocks = []
        
        # To speed up redundancy check later, we can maintain counts here too?
        # No, let's just build first, then check redundancy.
        
        while uncovered:
            if time.time() > deadline:
                break
                
            # Pick a random uncovered subset to start the block
            # This helps escape local minima compared to always picking min()
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                best_gain = -1
                best_point = None
                best_candidates = []
                
                for x in range(v):
                    if x in block:
                        continue
                    
                    # Gain: number of uncovered t-subsets that would be covered by adding x
                    # A t-subset U is covered if U is a subset of block U {x}
                    # This implies U \ {x} is a subset of block
                    # Since U has size t, and x is in U, U \ {x} has size t-1
                    # So we check if any (t-1)-subset of block, combined with x, is in uncovered
                    
                    gain = 0
                    # Generate all (t-1) subsets of block
                    for rest in itertools.combinations(block, t - 1):
                        # Create the candidate t-subset
                        # rest is a tuple of t-1 elements
                        # We need to check if sorted(rest + (x,)) is in uncovered
                        # Optimization: sort rest? rest is already sorted if block is sorted?
                        # block is not necessarily sorted during construction.
                        # But we can just create a frozenset or tuple.
                        # For speed, let's assume block is kept sorted? 
                        # No, just use a helper.
                        
                        # Create tuple of sorted elements
                        # Since t is small (5), this is cheap.
                        # rest is a tuple.
                        cand = tuple(sorted(rest + (x,)))
                        if cand in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                # Random tie-breaking
                if best_point is not None:
                    block.append(random.choice(best_candidates))
                else:
                    # No point adds any new coverage? 
                    # This shouldn't happen if uncovered is not empty, 
                    # unless the start_subset logic failed? 
                    # Actually, if we pick start_subset from uncovered, 
                    # and we build a block containing it, the first step 
                    # (adding the 4th point? no, start_subset is size t=5)
                    # Wait, start_subset is size 5. Block starts with 5 points.
                    # We need to add k - t = 8 - 5 = 3 more points.
                    # So the loop runs 3 times.
                    # If gain is 0 for all, we pick randomly.
                    block.append(random.choice([x for x in range(v) if x not in block]))
            
            # Sort block for consistency
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
        
        # Post-processing: Remove redundant blocks
        # Maintain count of coverage for each t-subset
        # count[sub] = number of blocks in 'blocks' that contain sub
        
        # Initialize counts
        counts = {}
        for b in blocks:
            for sub in itertools.combinations(b, t):
                counts[sub] = counts.get(sub, 0) + 1
        
        # Iteratively remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b in blocks:
                # Check if b is redundant
                redundant = True
                for sub in itertools.combinations(b, t):
                    if counts.get(sub, 0) <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove b
                    for sub in itertools.combinations(b, t):
                        counts[sub] -= 1
                    changed = True
                else:
                    new_blocks.append(b)
            
            blocks = new_blocks
        
        # Check if this result is better
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
            
    return best_blocks
