import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Generate all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_id = {sub: i for i, sub in enumerate(all_t_subs)}
    n_t = len(all_t_subs)
    
    # Generate all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subs)
    
    # Build Incidence Matrix M (dense float32 for fast dot product)
    # M[i, j] = 1.0 if t_sub j is contained in k_sub i
    M = np.zeros((n_k, n_t), dtype=np.float32)
    
    for i, b in enumerate(all_k_subs):
        for t_sub in itertools.combinations(b, t):
            j = t_sub_to_id[t_sub]
            M[i, j] = 1.0
            
    # Pre-compute boolean masks for fast update
    masks = (M > 0.5)
    
    # Initialize uncovered counts (1.0 if uncovered)
    uncovered = np.ones(n_t, dtype=np.float32)
    blocks = []
    
    while True:
        # Check time limit
        if time.time() > deadline:
            break
            
        # Calculate gains: number of uncovered t-subsets covered by each block
        gains = M.dot(uncovered)
        
        # Find the block with the maximum gain
        best_idx = np.argmax(gains)
        max_gain = gains[best_idx]
        
        if max_gain < 1e-6:
            # No gain, stop
            break
            
        # Add block to solution
        blocks.append(list(all_k_subs[best_idx]))
        
        # Update uncovered status
        uncovered[masks[best_idx]] = 0.0
        
        # Check if all covered
        if np.all(uncovered == 0.0):
            break
            
    return blocks
