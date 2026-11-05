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
    # block_triple_indices[i, j] is the index of the j-th triple in block i
    block_triple_indices = np.empty((n_blocks, k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, sub in enumerate(itertools.combinations(block, t)):
            block_triple_indices[i, j] = triple_to_idx[sub]
            
    def greedy_cover_ratio(random_seed):
        rng = np.random.RandomState(random_seed)
        covered_mask = np.zeros(n_triples, dtype=bool)
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains and overlaps
            # covered_mask[block_triple_indices] is (n_blocks, k_t) boolean
            block_covered = covered_mask[block_triple_indices]
            
            # Overlap: number of triples in block already covered
            overlaps = np.sum(block_covered, axis=1)
            
            # Gain: number of triples in block NOT covered
            gains = k_t - overlaps
            
            # If no gain, break (should not happen if remaining > 0 and logic is correct, 
            # but good for safety)
            if np.max(gains) == 0:
                break
                
            # Calculate ratio: Gain / (Overlap + 1)
            # Adding 1 to denominator avoids division by zero and ensures blocks with 
            # 0 overlap are highly preferred.
            ratios = gains / (overlaps + 1.0)
            
            # To add stochasticity and avoid deterministic local minima, 
            # we can add a small random noise to the ratios or use softmax sampling.
            # Let's use a simple method: pick the best, but with a small probability 
            # pick a random block from the top 10% to explore.
            # Or simpler: add small random noise to ratios.
            noise = rng.uniform(0, 0.1, size=n_blocks)
            scores = ratios + noise
            
            best_idx = np.argmax(scores)
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            new_covered = ~covered_mask[block_idx]
            remaining -= np.sum(new_covered)
            covered_mask[block_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))
            
        return selected_blocks

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    # Use more restarts as the heuristic is slightly more complex
    n_restarts = 50
    
    for i in range(n_restarts):
        blocks = greedy_cover_ratio(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
