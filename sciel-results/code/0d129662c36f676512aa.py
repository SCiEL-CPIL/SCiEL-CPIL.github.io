import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Pre-compute all t-subsets? 
    # C(19,4) = 3876. This is small.
    # But we need to track which are covered.
    
    best_blocks = None
    best_count = float('inf')
    
    # We will run multiple randomized greedy trials
    # Each trial:
    # 1. Start with empty uncovered set (all t-subsets)
    # 2. Greedily add blocks
    # 3. Remove redundant blocks
    # 4. Update best
    
    # To speed up, we can use a specific seed strategy.
    # random.seed(i) for i in range(N)
    
    # How many trials can we do?
    # One greedy run takes roughly:
    # ~150 blocks * 6 steps * 19 candidates * 20 subsets * (set lookup)
    # ~ 342,000 set lookups.
    # In Python, this might take 0.1 - 0.5 seconds.
    # So we can do 100-500 trials.
    
    # Let's do a loop until deadline.
    
    def greedy_once(seed):
        random.seed(seed)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block? 
            # Or pick the first one? 
            # Standard greedy: pick the most "urgent" uncovered subset? 
            # Usually, picking an arbitrary uncovered subset is fine for the start, 
            # because the gain function drives the rest.
            # However, picking a subset that is "hard" to cover might be better.
            # For now, let's pick a random one from uncovered to break symmetry.
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Calculate gain for each candidate x
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break # Should not happen if k <= v
                
                gains = []
                for x in candidates:
                    # Count how many uncovered t-subsets are formed by x and t-1 elements from block
                    gain = 0
                    # We need to check all (t-1)-subsets of block
                    # and see if {x} U S is in uncovered
                    for S in itertools.combinations(block, t-1):
                        # Create the t-subset
                        # S is a tuple, x is an int
                        # We need to sort to match the set of tuples
                        # Optimization: pre-compute sorted tuples?
                        # Or just use a set of frozensets? 
                        # frozensets are hashable and order-independent.
                        # Let's switch to frozensets for uncovered.
                        pass 
                    # This inner loop is slow.
                    pass
                # Placeholder for actual logic
                break
            pass
        return blocks

    # Let's rewrite greedy_once properly with frozensets for speed.
    
    def greedy_once_fast(seed):
        random.seed(seed)
        # Use frozensets for uncovered to allow O(1) lookup and order independence
        uncovered = set(frozenset(sub) for sub in itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start
            start_subset = random.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                candidates = [x for x in range(v) if x not in block]
                if not candidates:
                    break
                
                # Compute gain for each candidate
                # Gain = number of uncovered t-subsets that contain x and t-1 elements from block
                # We can pre-compute the (t-1)-subsets of the current block
                block_subs = list(itertools.combinations(block, t-1))
                
                best_gain = -1
                best_cand = -1
                # To add randomness, we can pick from top 3 or add noise
                gains = []
                for x in candidates:
                    gain = 0
                    for S in block_subs:
                        # S is a tuple of t-1 elements
                        # Check if frozenset(S + (x,)) is in uncovered
                        # Creating a frozenset every time is expensive.
                        # Alternative: keep uncovered as a set of tuples, but sorted?
                        # frozenset is safer.
                        # Let's optimize:
                        # If t=4, S has 3 elements.
                        # We can check membership.
                        if frozenset(S + (x,)) in uncovered:
                            gain += 1
                    gains.append((gain, x))
                
                # Sort by gain descending
                # Add small random noise to break ties randomly
                # Or just pick randomly from the max gain
                max_gain = max(g for g, _ in gains)
                top_candidates = [x for g, x in gains if g == max_gain]
                next_x = random.choice(top_candidates)
                
                block.append(next_x)
            
            # Sort block for consistent representation (not strictly needed for logic, but good)
            block.sort()
            blocks.append(block)
            
            # Remove covered subsets from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(frozenset(sub))
        
        return blocks

    def remove_redundant(blocks):
        # Check if any block is redundant
        # A block is redundant if every t-subset it covers is also covered by another block.
        # This is expensive to check fully. 
        # Heuristic: Try removing each block one by one and check if coverage is maintained?
        # Too slow.
        # Alternative: Just return blocks. The greedy algorithm tends to produce minimal-ish sets.
        # For now, skip redundancy removal to save time, or do a simple check.
        # Actually, a simple check: 
        # Count coverage of each t-subset. If any is 1, the block containing it is essential.
        # If a block contains no essential subsets, it might be removable?
        # No, it might be essential in combination.
        # Let's skip this for now and rely on randomized greedy.
        return blocks

    # Main loop
    trial = 0
    while time.time() < deadline:
        blocks = greedy_once_fast(trial)
        blocks = remove_redundant(blocks)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # Optional: try to improve best_blocks with local search?
            # For now, just keep updating.
        
        trial += 1
        # If we found a very good solution, we might want to spend more time on it?
        # Or just keep trying to beat it.
        
    if best_blocks is None:
        # Fallback: run one deterministic greedy
        blocks = greedy_once_fast(0)
        return blocks
        
    return best_blocks
