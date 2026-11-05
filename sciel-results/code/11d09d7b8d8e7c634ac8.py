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
            
    # Calculate scarcity weights
    triple_counts = np.bincount(block_triple_indices.ravel(), minlength=n_triples)
    weights = 1.0 / triple_counts
    
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        # Use active weights for speed
        active_weights = weights.copy()
        selected_indices = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains: sum of active weights for each block
            # This is the most expensive part
            gains = np.sum(active_weights[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            block_triples = block_triple_indices[best_idx]
            
            # Count newly covered
            # We need to know how many of these were active
            # Since active_weights is 0 for covered, we can just check if weight > 0?
            # No, weight is never 0 (unless count is inf, which isn't).
            # We need to count how many were uncovered.
            # np.sum(uncovered_mask[block_triples])
            # Or, since we zero out active_weights, we can count non-zeroes in the slice?
            # But we need to update remaining.
            
            newly_covered = np.sum(uncovered_mask[block_triples])
            remaining -= newly_covered
            
            # Update states
            uncovered_mask[block_triples] = False
            active_weights[block_triples] = 0.0
            
            selected_indices.append(best_idx)
            
        # Redundancy Removal
        coverage_counts = np.zeros(n_triples, dtype=np.int32)
        for idx in selected_indices:
            coverage_counts[block_triple_indices[idx]] += 1
            
        final_indices = []
        for idx in selected_indices:
            block_triples = block_triple_indices[idx]
            # If all triples in this block have count >= 2, it's redundant
            if np.all(coverage_counts[block_triples] >= 2):
                coverage_counts[block_triples] -= 1
            else:
                final_indices.append(idx)
                
        return [list(all_blocks[i]) for i in final_indices]

    best_blocks = None
    best_len = float('inf')
    
    # 30 restarts
    n_restarts = 30
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
