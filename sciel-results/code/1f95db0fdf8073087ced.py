import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    # Pre-compute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Pre-compute all k-blocks and the t-subsets they cover
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # block_subsets[i] is a list of indices of t-subsets covered by k_blocks[i]
    # Shape: (num_k_blocks, C(k,t))
    block_subsets = np.zeros((num_k_blocks, int(np.prod([1])) if k < t else 1), dtype=np.int32) 
    # Actually, C(k,t) is fixed. Let's compute it properly.
    # C(7,4) = 35.
    # We can just build a list of lists first, then convert to array.
    block_subsets_list = []
    for block in k_blocks:
        covered = [t_subset_to_idx[subset] for subset in itertools.combinations(block, t)]
        block_subsets_list.append(covered)
    
    # Convert to a 2D numpy array. All lists have same length C(k,t).
    block_subsets = np.array(block_subsets_list, dtype=np.int32)
    
    # State
    coverage = np.zeros(num_t_subsets, dtype=np.int32)
    used_blocks = np.zeros(num_k_blocks, dtype=bool)
    selected_blocks = []
    
    # Greedy Loop
    while True:
        # Check time
        if time.time() > deadline:
            break
            
        # Calculate gains: number of uncovered t-subsets covered by each block
        uncovered_mask = (coverage == 0)
        
        # For each block, count how many of its t-subsets are currently uncovered
        # block_subsets shape: (N, M)
        # uncovered_mask[block_subsets] shape: (N, M)
        # sum along axis 1 gives gain for each block
        gains = np.sum(uncovered_mask[block_subsets], axis=1)
        
        # Mask out already used blocks
        gains[used_blocks] = -1
        
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            # All t-subsets are covered
            break
            
        # Select the block
        selected_blocks.append(k_blocks[best_idx])
        used_blocks[best_idx] = True
        
        # Update coverage
        # Increment coverage for all t-subsets covered by this block
        # We need to be careful not to double count if we just do += 1
        # But coverage is a count, so += 1 is correct.
        block_t_indices = block_subsets[best_idx]
        coverage[block_t_indices] += 1
        
        # Optional: Early exit if all covered
        # if not np.any(coverage == 0):
        #     break
        
    return [list(b) for b in selected_blocks]
