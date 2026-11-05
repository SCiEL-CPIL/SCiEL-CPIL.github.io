"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Precompute all t-subsets and map them to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    total_t_subsets = len(all_t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Precompute (t-1)-subsets for faster gain calculation?
    # Gain of adding x to block B:
    # Number of (t-1)-subsets S of B such that S U {x} is uncovered.
    
    best_blocks = None
    best_count = float('inf')
    
    num_restarts = 0
    
    while time.time() - start_time < time_limit:
        num_restarts += 1
        # Randomized Greedy
        # uncovered_count[i] = 1 if subset i is uncovered, 0 otherwise
        uncovered_count = [1] * total_t_subsets
        uncovered_set = set(range(total_t_subsets))
        
        blocks = []
        
        while uncovered_set:
            # Pick a random uncovered subset to start the block
            # Or pick the one with highest "demand"? 
            # Standard greedy: pick arbitrary uncovered, then maximize gain.
            # Randomized: pick random uncovered to allow different paths.
            current_uncovered_list = list(uncovered_set)
            if not current_uncovered_list:
                break
            target_idx = random.choice(current_uncovered_list)
            target_subset = all_t_subsets[target_idx]
            
            block = list(target_subset)
            block_set = set(block)
            
            # Expand block to size k
            while len(block) < k:
                best_gain = -1
                best_point = -1
                # Precompute (t-1)-subsets of current block
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in block_t_minus_1:
                        # Check if sub + x is a valid t-subset and uncovered
                        # We need to check if the sorted tuple is in subset_to_idx
                        # And if its index is in uncovered_set
                        candidate = tuple(sorted(sub + (x,)))
                        if candidate in subset_to_idx:
                            idx = subset_to_idx[candidate]
                            if idx in uncovered_set:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = x
                
                if best_point == -1:
                    # Should not happen if v > k
                    break
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered set
            # For all t-subsets in this block, remove from uncovered
            for sub in itertools.combinations(block, t):
                idx = subset_to_idx[sub]
                if idx in uncovered_set:
                    uncovered_set.remove(idx)
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            # If we reach a very good number, we can stop early? 
            # Lower bound is 131. Best known is 151.
            # If we get close to 150, we might still want to try to improve.
            # But time is limited.
    
    if best_blocks is None:
        # Fallback to simple greedy if something went wrong
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks
        
    return best_blocks
# EVOLVE-BLOCK-END
