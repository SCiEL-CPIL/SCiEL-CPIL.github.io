import itertools
import time
import random
from math import comb

# Precomputation for v=19, k=7, t=4
# However, the function signature is construct(v, k, t).
# We should handle general v, k, t but optimize for the specific case if possible.
# The task says "The evaluator passes this task's parameters v, k and t".
# So we can assume v=19, k=7, t=4 for optimization, but keep it general enough.
# Given the 120s limit, precomputing for v=19 is fast.

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    
    # Map each t-subset to an index
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    
    # Precompute the coverage bitmask for each k-subset
    # A block covers a t-subset if the t-subset is a subset of the block.
    block_masks = []
    for block in k_subsets:
        mask = 0
        # Check all t-subsets of this block
        # Since k=7, t=4, there are C(7,4)=35 t-subsets per block.
        # It's faster to iterate over t-subsets of the block directly.
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
    
    # Initialize uncovered set
    uncovered = (1 << num_t_subsets) - 1
    
    blocks = []
    
    # Greedy loop
    while uncovered:
        if time.time() > deadline:
            break
            
        best_gain = -1
        best_block_idx = -1
        candidates = []
        
        # Find the block with maximum new coverage
        # Optimization: If num_t_subsets is large, iterating all blocks is O(C(v,k)).
        # For v=19, k=7, C(19,7)=50388. This is small enough to iterate fully each step.
        # Number of steps is at most C(v,t) = 3876, but typically much less (~150).
        # 150 * 50388 = ~7.5 million operations. Very fast in Python if optimized.
        
        for i, mask in enumerate(block_masks):
            # Count how many uncovered t-subsets this block covers
            # We only care about bits that are set in both mask and uncovered
            new_coverage = mask & uncovered
            if new_coverage:
                # popcount
                gain = bin(new_coverage).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
                    
        if best_block_idx == -1:
            # Should not happen if uncovered is not empty, but safe check
            break
            
        # Tie-breaking: Pick randomly from top candidates to explore different paths
        # Or pick the one that minimizes overlap with already selected blocks?
        # For now, random among top candidates helps avoid deterministic traps.
        
        chosen_idx = random.choice(candidates)
        blocks.append(k_subsets[chosen_idx])
        
        # Update uncovered
        uncovered &= ~block_masks[chosen_idx]
        
    return blocks
