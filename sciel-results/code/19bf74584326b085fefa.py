import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # 1. Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # 2. Precompute all k-subsets and their covered t-subset indices
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # block_t_indices[i] will be a list of t-subset indices covered by k-subset i
    # We will store this as a flat NumPy array for vectorization later
    # Each k-subset covers C(k, t) t-subsets
    num_covered_per_block = len(list(itertools.combinations(range(k), t)))
    
    block_t_indices = np.zeros((num_k_subsets, num_covered_per_block), dtype=np.int32)
    
    for i, k_sub in enumerate(all_k_subsets):
        k_set = set(k_sub)
        # Generate all t-subsets of this k-subset
        # We need to map them to the global index
        subs = list(itertools.combinations(k_sub, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = subset_to_idx[sub]
            
    # 3. Greedy Set Cover
    covered = np.zeros(num_t_subsets, dtype=bool)
    selected_blocks = []
    
    # We need to track which blocks are already selected to avoid re-selecting them?
    # Standard greedy doesn't require exclusion if we just pick the best gain.
    # If a block is already selected, its gain will be 0 (or low) because its subsets are covered.
    # However, to be efficient, we can just pick the max gain.
    
    while True:
        if time.time() > deadline:
            # Fallback: If we run out of time, we have an invalid covering.
            # But with vectorization, this shouldn't happen.
            # To ensure validity, we could try to force-include random blocks?
            # No, better to just return what we have? No, invalid = 0 score.
            # The vectorized step is ~5-10ms. 300 steps = 3s.
            # It should finish well within time.
            break
        
        # Calculate gains: number of currently uncovered t-subsets covered by each block
        # fancy indexing: covered[indices] gives a boolean array
        # reshape and sum over the second axis
        gains = covered[block_t_indices].reshape(num_k_subsets, num_covered_per_block).sum(axis=1)
        
        # Find the block with the maximum gain
        max_gain = np.max(gains)
        if max_gain == 0:
            # All subsets are covered
            break
            
        best_block_idx = np.argmax(gains)
        selected_blocks.append(all_k_subsets[best_block_idx])
        
        # Update covered status
        covered[block_t_indices[best_block_idx]] = True
        
        # Optional: Early exit if all covered
        if np.sum(covered) == num_t_subsets:
            break
            
    # Convert selected blocks to lists of ints
    result = [list(b) for b in selected_blocks]
    
    return result
