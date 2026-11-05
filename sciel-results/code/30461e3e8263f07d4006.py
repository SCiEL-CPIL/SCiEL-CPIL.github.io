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
            
    def greedy_cover(random_seed):
        # Use a local RandomState to avoid global side effects
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        cover_counts = np.zeros(n_triples, dtype=np.int32)
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains: number of uncovered triples in each block
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate to break ties randomly
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_triples = block_triple_indices[best_idx]
            uncovered_mask[block_triples] = False
            cover_counts[block_triples] += 1
            remaining -= max_gain
            selected_blocks.append(best_idx)
            
        # Local Search: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            # Iterate backwards to remove recently added blocks first
            for i in range(len(selected_blocks) - 1, -1, -1):
                b_idx = selected_blocks[i]
                triples = block_triple_indices[b_idx]
                # If all triples in this block are covered by at least one other block
                if np.min(cover_counts[triples]) > 1:
                    cover_counts[triples] -= 1
                    selected_blocks.pop(i)
                    changed = True
                    
        return [list(all_blocks[b]) for b in selected_blocks]

    best_blocks = None
    best_len = float('inf')
    
    # Increase restarts to improve quality
    n_restarts = 100
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
