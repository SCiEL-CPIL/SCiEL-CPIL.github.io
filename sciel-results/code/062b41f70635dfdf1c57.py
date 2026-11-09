import numpy as np
from itertools import combinations
import random
import time

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build mapping from k-subset to its t-subset indices
    # C(k, t) is the number of t-subsets in a k-set
    n_t_in_k = len(list(combinations(range(k), t)))
    block_t_indices = np.zeros((n_k, n_t_in_k), dtype=np.int32)
    for i, ks in enumerate(k_subs):
        for j, ts in enumerate(combinations(ks, t)):
            block_t_indices[i, j] = t_sub_idx[ts]
            
    best_blocks = None
    best_len = float('inf')
    
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Number of iterations to try
    # We will try as many as possible
    iteration = 0
    
    while time.time() - start_time < max_time:
        iteration += 1
        # 1. Randomized Construction
        uncovered = np.ones(n_t, dtype=bool)
        n_uncovered = n_t
        blocks = []
        selected_set = set()
        
        # Randomly pick blocks until all t-subsets are covered
        # To speed up, we can pick a random block and check gain
        # Pure random is slow if it picks many useless blocks.
        # Let's use a slight bias: Pick random, but if we have many blocks,
        # the probability of picking a new one is high anyway.
        
        while n_uncovered > 0:
            # Pick random index
            idx = np.random.randint(n_k)
            
            if idx in selected_set:
                continue
                
            # Calculate gain
            t_idx = block_t_indices[idx]
            # Check how many of these are currently uncovered
            # gain = np.sum(uncovered[t_idx])
            # If gain == 0, skip? 
            # In random construction, we add it anyway if we want to ensure termination,
            # but skipping gain=0 is better for minimization.
            
            # However, if we skip gain=0, we might loop forever if we are unlucky?
            # No, eventually we will pick one with gain > 0.
            
            # Optimization: Only add if gain > 0
            # This is effectively "Randomized Greedy" where we filter out zero-gain blocks.
            # This is distinct from "Deterministic Greedy" which picks MAX gain.
            
            gain = np.sum(uncovered[t_idx])
            if gain == 0:
                continue
                
            # Add block
            blocks.append(idx)
            selected_set.add(idx)
            uncovered[t_idx] = False
            n_uncovered -= gain
            
        # 2. Local Search (Deletion)
        # Count coverage
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b_idx in blocks:
            cover_count[block_t_indices[b_idx]] += 1
            
        improved = True
        while improved:
            improved = False
            new_blocks = []
            for b_idx in blocks:
                t_idx = block_t_indices[b_idx]
                # If min cover count for this block's t-subsets is > 1, we can remove it
                if np.min(cover_count[t_idx]) > 1:
                    cover_count[t_idx] -= 1
                    improved = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
        # Update best
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
        # Optional: If we found a very good solution, we can break early?
        # Or just keep trying.
        
    # Convert indices to actual blocks
    if best_blocks is None:
        # Fallback: Return a valid covering (should not happen)
        return [list(k_subs[i]) for i in range(n_k)]
        
    return [list(k_subs[i]) for i in best_blocks]
