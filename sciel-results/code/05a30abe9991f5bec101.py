import itertools
import numpy as np
from math import comb

def construct(v, k, t):
    # 1. Precomputation
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {trip: i for i, trip in enumerate(all_triples)}
    n_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    k_t = comb(k, t)
    
    # Store block indices for fast access
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]

    def stochastic_greedy(random_seed, alpha=1.0):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            # Filter out blocks with 0 gain
            valid_indices = np.where(gains > 0)[0]
            if len(valid_indices) == 0:
                break
                
            valid_gains = gains[valid_indices]
            
            # Apply power transformation to control exploration
            # alpha < 1 makes it more random, alpha > 1 makes it more greedy
            weights = valid_gains ** alpha
            total_weight = np.sum(weights)
            
            if total_weight == 0:
                break
                
            probs = weights / total_weight
            
            # Sample an index
            chosen_pos = rng.choice(len(valid_indices), p=probs)
            best_idx = valid_indices[chosen_pos]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            new_covered = np.sum(uncovered_mask[block_idx])
            uncovered_mask[block_idx] = False
            remaining -= new_covered
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 100
    
    for i in range(n_restarts):
        # Vary alpha slightly? Or keep fixed?
        # Let's keep alpha=1.0 for now, which is proportional to gain.
        # If this doesn't improve, we might need to tune alpha.
        blocks = stochastic_greedy(i, alpha=1.0)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
