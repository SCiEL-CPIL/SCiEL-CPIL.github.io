import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 10 seconds before the 150s kill switch
    deadline = start_time + 140.0
    
    # Handle trivial cases
    if t == 1:
        # Each point needs to be in a block.
        # We can put k points in each block.
        # Number of blocks = ceil(v/k)
        # But we need to return the blocks.
        # This is not the target case, but good for robustness.
        blocks = []
        remaining = list(range(v))
        while remaining:
            block = remaining[:k]
            blocks.append(sorted(block))
            remaining = remaining[k:]
        return blocks

    best_blocks = None
    best_count = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        uncovered = set(itertools.combinations(range(v), t))
        if not uncovered:
            return []
        
        blocks = []
        while uncovered:
            # Pick a random uncovered t-subset to start the block
            # Converting to list is O(N), but N is small (3876)
            start_sub = random.choice(list(uncovered))
            block = list(start_sub)
            
            while len(block) < k:
                best_gain = -1
                best_candidates = []
                
                # Pre-compute combinations of current block of size t-1
                # This is constant for this step
                t_minus_1 = t - 1
                block_combos = list(itertools.combinations(block, t_minus_1))
                
                # Iterate over all points not in the block
                for x in range(v):
                    if x in block:
                        continue
                    
                    gain = 0
                    # Check how many t-subsets formed by x and a combo from block
                    # are currently uncovered
                    for combo in block_combos:
                        # Create the t-subset
                        # x is an int, combo is a tuple of ints
                        # We need a sorted tuple to match the set keys
                        sub = tuple(sorted((x,) + combo))
                        if sub in uncovered:
                            gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_candidates = [x]
                    elif gain == best_gain:
                        best_candidates.append(x)
                
                # If best_gain is 0, it means no point adds new coverage.
                # This can happen if the block is "complete" in terms of covering
                # the specific uncovered sets it can touch, or if we are stuck.
                # We must still fill the block to size k.
                # Pick a random candidate to break ties or if all gains are 0.
                if best_candidates:
                    block.append(random.choice(best_candidates))
                else:
                    # Fallback: pick any point not in block
                    # This should rarely happen
                    remaining = [x for x in range(v) if x not in block]
                    if remaining:
                        block.append(random.choice(remaining))
                    else:
                        break # Should not happen if len(block) < k
            
            block.sort()
            blocks.append(block)
            
            # Remove all t-subsets contained in this block from uncovered
            for sub in itertools.combinations(block, t):
                uncovered.discard(sub)
                
        return blocks

    # Run initial greedy
    current_blocks = run_greedy(42)
    if len(current_blocks) < best_count:
        best_blocks = current_blocks
        best_count = len(current_blocks)
        
    # Try to improve with more restarts
    seed = 43
    while time.time() < deadline:
        # Check time
        if time.time() > deadline:
            break
            
        blocks = run_greedy(seed)
        seed += 1
        
        if len(blocks) < best_count:
            best_blocks = blocks
            best_count = len(blocks)
            
    # If no blocks were found (should not happen), return empty or fallback
    if best_blocks is None:
        return []
        
    return best_blocks
