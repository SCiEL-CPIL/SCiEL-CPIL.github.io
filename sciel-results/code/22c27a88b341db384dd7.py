import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    timeout = 115  # Safety margin for 120s limit
    
    # Precompute all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(k_blocks)
    
    # Precompute t-subsets for a generic block of size k
    # These are relative indices
    t_subsets_rel = list(itertools.combinations(range(k), t))
    num_t_per_block = len(t_subsets_rel)
    
    # Map relative t-subsets to absolute indices for each block
    # block_t_indices[i] contains the indices of t-subsets in block i
    block_t_indices = np.empty((n_k_blocks, num_t_per_block), dtype=np.int32)
    
    for i, block in enumerate(k_blocks):
        for j, ts_rel in enumerate(t_subsets_rel):
            abs_ts = tuple(block[rel] for rel in ts_rel)
            block_t_indices[i, j] = t_subset_index[abs_ts]
            
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts with random tie-breaking
    max_restarts = 50 # Will be limited by timeout
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
            
        covered = np.zeros(n_t_subsets, dtype=bool)
        blocks = []
        
        while not np.all(covered):
            if time.time() - start_time > timeout:
                break
                
            # Calculate gains
            gains = np.sum(~covered[block_t_indices], axis=1)
            
            # Add small random noise to break ties and diversify search
            # The noise should be small enough not to override significant gain differences
            # but large enough to randomize choices among equal-gain blocks
            gains += np.random.uniform(0, 0.001, size=n_k_blocks)
            
            if np.max(gains) < 0.0005: # Threshold to ensure we only pick blocks with real gain
                break
                
            best_block_idx = np.argmax(gains)
            
            # Add the block
            blocks.append(list(k_blocks[best_block_idx]))
            
            # Update covered
            covered[block_t_indices[best_block_idx]] = True
            
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
    return best_blocks
