import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110  # seconds
    
    best_blocks = None
    best_size = float('inf')
    
    # Run multiple random restarts
    while time.time() - start_time < time_limit:
        rng = random.Random()
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # Using min(uncovered) is deterministic. Randomizing helps escape local minima.
            # However, picking the "first" one in a sorted order is often good for locality.
            # Let's try random selection.
            start_subset = rng.choice(list(uncovered))
            block = list(start_subset)
            
            while len(block) < k:
                # Calculate gain for each candidate point
                candidates = [x for x in range(v) if x not in block]
                # For each candidate, count how many new t-subsets it covers
                # A t-subset is covered if it contains (block_subset of size t-1) and x
                # and is currently in 'uncovered'
                
                gains = {}
                for x in candidates:
                    # Generate all (t-1)-subsets of current block
                    # If t-1 > len(block), gain is 0
                    if len(block) < t - 1:
                        gains[x] = 0
                    else:
                        count = 0
                        for rest in itertools.combinations(block, t - 1):
                            # Create the t-subset
                            t_sub = tuple(sorted(rest + (x,)))
                            if t_sub in uncovered:
                                count += 1
                        gains[x] = count
                
                # Find the best candidate
                # To break ties randomly, we can add a small random noise or sort by (gain, random)
                # Let's sort by gain descending, then random key
                best_x = max(candidates, key=lambda x: (gains[x], rng.random()))
                block.append(best_x)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for t_sub in itertools.combinations(block, t):
                uncovered.discard(t_sub)
        
        if len(blocks) < best_size:
            best_size = len(blocks)
            best_blocks = blocks
            # Optional: If we find a very good solution (e.g., <= 151), we can stop early
            if best_size <= 151:
                break
                
    return best_blocks
