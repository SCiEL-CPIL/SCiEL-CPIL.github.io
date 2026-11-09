import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Precompute all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    
    # Map each t-subset to an index
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-subsets and map to indices
    k_subsets = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(k_subsets)}
    
    # For each k-subset, precompute the list of t-subset indices it covers
    block_covered_t_indices = []
    for block in k_subsets:
        covered = []
        for t_sub in itertools.combinations(block, t):
            covered.append(t_sub_to_idx[t_sub])
        block_covered_t_indices.append(np.array(covered, dtype=np.int64))
    
    # uncovered_mask: numpy array of bools, True if t-subset is uncovered
    uncovered_mask = np.ones(num_t, dtype=bool)
    
    blocks = []
    
    while uncovered_mask.any():
        # Deadline guard
        if time.time() - start_time > deadline:
            break
        
        # Find an uncovered t-subset (first one)
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        s_idx = uncovered_indices[0]
        s_points = list(t_subsets[s_idx])
        s_set = set(s_points)
        remaining_points = [p for p in range(v) if p not in s_set]
        
        best_block_points = None
        best_gain = -1
        
        # Generate all k-subsets containing s_points
        for extra in itertools.combinations(remaining_points, k - t):
            b_points = tuple(sorted(s_points + list(extra)))
            
            # Calculate gain using precomputed coverage
            b_idx = block_to_idx[b_points]
            covered_idx = block_covered_t_indices[b_idx]
            gain = np.sum(uncovered_mask[covered_idx])
            
            if gain > best_gain:
                best_gain = gain
                best_block_points = b_points
                
        if best_block_points is not None:
            blocks.append(list(best_block_points))
            # Update uncovered_mask
            b_idx = block_to_idx[best_block_points]
            uncovered_mask[block_covered_t_indices[b_idx]] = False
            
    return blocks
