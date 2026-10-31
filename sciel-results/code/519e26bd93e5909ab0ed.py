"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
from math import comb


def construct(v, k, t):
    deadline = time.time() + 115.0
    total_t_subsets = comb(v, t)
    
    # Precompute all t-subsets as integers using combinatorial number system
    def subset_to_int(sub):
        # Convert a sorted tuple of length t to a unique integer
        # Using combinatorial number system
        result = 0
        n = v
        for i, val in enumerate(sub):
            # C(val, i)
            result += comb(val, i)
        return result
    
    # Generate all t-subsets and their integer representations
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Uncovered as a boolean array
    uncovered = np.ones(total_t_subsets, dtype=np.int32)
    remaining = total_t_subsets
    blocks = []
    
    # Precompute: for each point x and each (t-1)-subset, what t-subsets does x+subset form?
    # Actually, let's precompute for each point x, the list of t-subsets containing x
    # This is C(v-1, t-1) per point
    
    # For speed: precompute the index of each t-subset
    # For a given block B (sorted list of points), the t-subsets of B are C(k,t)
    # We need to quickly find which of those are uncovered
    
    # Strategy: for each candidate point x not in current block,
    # compute gain = number of uncovered t-subsets that would be covered by adding x to block
    # A t-subset is covered by adding x to block if:
    #   - it contains x, AND
    #   - the other t-1 elements are in the current block
    
    # Precompute: for each point x, list of all t-subsets containing x
    # t_subsets_with[x] = list of indices of t-subsets that contain x
    t_subsets_with = [[] for _ in range(v)]
    for idx, sub in enumerate(all_t_subsets):
        for pt in sub:
            t_subsets_with[pt].append(idx)
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Pick the first uncovered t-subset to start the block
        # Find first uncovered
        uncov_indices = np.where(uncovered > 0)[0]
        if len(uncov_indices) == 0:
            break
        start_idx = uncov_indices[0]
        start_sub = all_t_subsets[start_idx]
        block = list(start_sub)
        
        # Extend block to size k
        while len(block) < k:
            if time.time() > deadline:
                break
            
            block_set = set(block)
            best_x = -1
            best_gain = -1
            
            # For each candidate x not in block, compute gain
            # Gain = number of uncovered t-subsets that contain x and whose other t-1 elements are in block
            # Precompute: for each x, the t-subsets containing x
            # For each such t-subset, check if all other elements (besides x) are in block
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for idx in t_subsets_with[x]:
                    if uncovered[idx] == 0:
                        continue
                    sub = all_t_subsets[idx]
                    # Check if all elements except x are in block
                    ok = True
                    for pt in sub:
                        if pt != x and pt not in block_set:
                            ok = False
                            break
                    if ok:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x == -1:
                # No improvement possible, just pick any point not in block
                for x in range(v):
                    if x not in block_set:
                        best_x = x
                        break
                if best_x == -1:
                    break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        
        # Check if this block is already used
        block_tuple = tuple(block)
        if block_tuple in [tuple(b) for b in blocks]:
            # Skip duplicate (shouldn't happen often)
            continue
        
        blocks.append(block)
        
        # Update uncovered: mark all t-subsets of this block as covered
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[tuple(sub)]
            if uncovered[idx] > 0:
                uncovered[idx] = 0
                remaining -= 1
    
    return blocks
# EVOLVE-BLOCK-END
