"""Improved program for C(19,7,4) covering design."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
from collections import defaultdict


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    
    # Track coverage: which t-subsets are covered
    covered = [False] * total_t_subsets
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    blocks = []
    
    while True:
        # Check deadline
        if time.time() > deadline:
            break
        
        # Count uncovered
        uncovered_count = sum(1 for c in covered if not c)
        if uncovered_count == 0:
            break
        
        # Find an uncovered t-subset to start with (pick one with fewest options)
        # For efficiency, just pick the first uncovered one
        start_subset = None
        for i, s in enumerate(all_t_subsets):
            if not covered[i]:
                start_subset = s
                break
        
        if start_subset is None:
            break
        
        # Build a block greedily: start with start_subset, add points that cover most uncovered t-subsets
        block = list(start_subset)
        
        # Precompute: for each point, how many uncovered t-subsets containing it exist
        # We'll do this incrementally
        
        while len(block) < k:
            if time.time() > deadline:
                break
            
            # Find point that maximizes coverage of uncovered t-subsets
            best_point = -1
            best_gain = -1
            
            # For each candidate point, count how many uncovered t-subsets it would cover
            # A t-subset is covered by adding point x if:
            # 1. x is in the t-subset
            # 2. The other t-1 elements are in the current block
            # 3. The t-subset is currently uncovered
            
            block_set = set(block)
            
            # Group uncovered t-subsets by which points they contain
            # For efficiency, iterate over candidates
            for x in range(v):
                if x in block_set:
                    continue
                
                # Count uncovered t-subsets that contain x and whose other t-1 elements are in block
                gain = 0
                # We need t-subsets S where x in S, S \ {x} has t-1 elements all in block
                # And S is uncovered
                
                # Iterate over combinations of (t-1) elements from block
                for combo in itertools.combinations(block, t - 1):
                    # Check if combo + (x,) forms a valid t-subset
                    candidate = tuple(sorted(combo + (x,)))
                    if candidate in subset_to_idx:
                        idx = subset_to_idx[candidate]
                        if not covered[idx]:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point == -1 or best_gain == 0:
                # No improvement possible, pick a random point not in block
                available = [x for x in range(v) if x not in block_set]
                if available:
                    best_point = random.choice(available)
                else:
                    break
            
            block.append(best_point)
        
        if len(block) < k:
            # Couldn't complete the block, try to fill with any available points
            block_set = set(block)
            for x in range(v):
                if x not in block_set and len(block) < k:
                    block.append(x)
        
        if len(block) < k:
            break
        
        # Sort the block
        block.sort()
        blocks.append(block)
        
        # Update covered array
        for t_subset in itertools.combinations(block, t):
            idx = subset_to_idx[t_subset]
            covered[idx] = True
        
        # Check if we've covered everything
        if all(covered):
            break
    
    # Verify validity and return
    return blocks
# EVOLVE-BLOCK-END
