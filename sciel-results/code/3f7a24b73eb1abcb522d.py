import numpy as np
from itertools import combinations
from functools import reduce

def construct(v, k, t):
    # Precompute all t-subsets and map them to indices
    t_subs = list(combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute all k-subsets
    k_subs = list(combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build mapping from k-subset to its t-subset indices
    # For each k-subset, find the indices of all its t-subsets
    block_t_indices = np.zeros((n_k, len(list(combinations(range(k), t)))), dtype=np.int32)
    for i, ks in enumerate(k_subs):
        for j, ts in enumerate(combinations(ks, t)):
            block_t_indices[i, j] = t_sub_idx[ts]
            
    # Precompute frequency of each t-subset across all k-subsets
    # This helps in tie-breaking: prefer blocks that cover rarer t-subsets
    freq = np.bincount(block_t_indices.ravel(), minlength=n_t).astype(np.float64)
    freq_inv = 1.0 / (freq + 1e-9)

    # Uncovered mask
    uncovered = np.ones(n_t, dtype=bool)
    n_uncovered = n_t
    blocks = []
    
    # Greedy loop
    while n_uncovered > 0:
        # Compute gain for each block: number of currently uncovered t-subsets it covers
        gains = uncovered[block_t_indices].sum(axis=1)
        
        # Pick the block with maximum gain
        max_gain = gains.max()
        
        if max_gain == 0:
            break
            
        # Find all blocks with maximum gain
        candidates = np.where(gains == max_gain)[0]
        
        if len(candidates) == 1:
            best_idx = candidates[0]
        else:
            # Tie-breaking: choose the block that covers the "rarest" uncovered t-subsets
            cand_indices = block_t_indices[candidates]
            cand_uncovered_mask = uncovered[cand_indices]
            scores = np.sum(freq_inv[cand_indices] * cand_uncovered_mask, axis=1)
            best_cand_local_idx = np.argmax(scores)
            best_idx = candidates[best_cand_local_idx]
            
        blocks.append(k_subs[best_idx])
        
        # Update uncovered mask
        uncovered[block_t_indices[best_idx]] = False
        n_uncovered -= max_gain
    
    # Convert tuples to lists
    return [list(b) for b in blocks]
