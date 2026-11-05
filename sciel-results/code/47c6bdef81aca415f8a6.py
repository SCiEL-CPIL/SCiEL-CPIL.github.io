import itertools
import time
import random

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a randomized greedy algorithm.
    """
    deadline = time.time() + 140.0
    best_blocks = None
    best_len = float('inf')
    
    # Pre-compute all t-subsets as a set for O(1) lookup
    all_subsets = set(itertools.combinations(range(v), t))
    
    # Run multiple iterations of the greedy algorithm with randomization
    while time.time() < deadline:
        uncovered = set(all_subsets)
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start the block
            # Converting to list is O(N), but N decreases.
            # For N=3876, this is fast.
            seed = random.choice(list(uncovered))
            current_block = set(seed)
            
            # Fill the block to size k
            while len(current_block) < k:
                best_gain = -1
                best_x = None
                
                # Iterate through all points not in current_block
                for x in range(v):
                    if x in current_block:
                        continue
                    
                    candidate_block = current_block | {x}
                    gain = 0
                    
                    # Count how many uncovered subsets are contained in candidate_block
                    # We only need to check combinations of size t
                    for sub in itertools.combinations(candidate_block, t):
                        if sub in uncovered:
                            gain += 1
                            
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                
                if best_x is not None:
                    current_block.add(best_x)
                else:
                    # Fallback: should not happen as seed is uncovered
                    break
            
            # Add the block to our list
            blocks.append(tuple(sorted(current_block)))
            
            # Update uncovered set
            block_subsets = set(itertools.combinations(current_block, t))
            uncovered.difference_update(block_subsets)
            
        # End of one full covering attempt
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
    
    if best_blocks is None:
        return []
        
    # Convert tuples to lists as required by the interface
    return [list(b) for b in best_blocks]
