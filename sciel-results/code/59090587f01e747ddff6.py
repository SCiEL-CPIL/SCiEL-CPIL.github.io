import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    
    # 2. Generate all k-subsets and their contained t-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # Precompute block_t_indices: shape (num_k_subsets, C(k,t))
    # C(k,t) is the number of t-subsets in a k-subset
    c_k_t = 1
    for i in range(t):
        c_k_t = c_k_t * (k - i) // (i + 1)
        
    block_t_indices = np.zeros((num_k_subsets, c_k_t), dtype=np.int32)
    
    # Efficiently fill block_t_indices
    # We can iterate over k_subsets and find t-subsets
    for i, k_sub in enumerate(k_subsets):
        # Find all t-subsets of k_sub
        # Using itertools.combinations is fine for small k,t
        t_in_k = list(itertools.combinations(k_sub, t))
        for j, t_sub in enumerate(t_in_k):
            block_t_indices[i, j] = t_subset_to_idx[t_sub]
            
    # 3. Greedy Set Cover with Random Sampling and Redundancy Removal
    best_blocks = None
    best_count = float('inf')
    
    rng = np.random.default_rng(42)
    
    # Run multiple restarts
    while time.time() < deadline:
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        blocks_used = []
        remaining = num_t_subsets
        
        while remaining > 0:
            # Sample M blocks
            M = 5000
            if num_k_subsets < M:
                sample_indices = np.arange(num_k_subsets)
            else:
                sample_indices = rng.choice(num_k_subsets, M, replace=False)
                
            # Get t-subset indices for sampled blocks
            # Shape: (M, c_k_t)
            sample_block_t_indices = block_t_indices[sample_indices]
            
            # Calculate gain: number of t-subsets in each block that are currently uncovered
            gains = np.sum(cover_count[sample_block_t_indices] == 0, axis=1)
            
            # Find best block in sample
            max_gain_idx = np.argmax(gains)
            best_block_idx = sample_indices[max_gain_idx]
            max_gain = gains[max_gain_idx]
            
            if max_gain == 0:
                # Fallback: pick a random block that covers at least one uncovered subset
                # This shouldn't happen often, but if it does, we need to handle it.
                # We can just break and try another restart.
                break
                
            # Update cover_count
            cover_count[block_t_indices[best_block_idx]] += 1
            remaining -= max_gain
            blocks_used.append(best_block_idx)
            
        if remaining == 0:
            # Cover is valid
            # 4. Redundancy Removal
            changed = True
            while changed:
                changed = False
                new_blocks = []
                for b_idx in blocks_used:
                    indices = block_t_indices[b_idx]
                    # Check if all subsets are covered by at least one other block
                    if np.all(cover_count[indices] > 1):
                        cover_count[indices] -= 1
                        changed = True
                    else:
                        new_blocks.append(b_idx)
                blocks_used = new_blocks
                
            if len(blocks_used) < best_count:
                best_count = len(blocks_used)
                best_blocks = list(blocks_used)
                
    # Convert indices back to blocks
    if best_blocks is None:
        return []
        
    return [list(k_subsets[i]) for i in best_blocks]
