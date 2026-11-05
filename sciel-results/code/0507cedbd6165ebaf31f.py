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

    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_block_indices = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains: number of uncovered triples in each block
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_triples = block_triple_indices[best_idx]
            newly_covered = np.sum(uncovered_mask[block_triples])
            uncovered_mask[block_triples] = False
            remaining -= newly_covered
            selected_block_indices.append(best_idx)
            
        # Post-processing: Remove redundant blocks
        # A block is redundant if all its triples are covered by other blocks in the set.
        # We iterate and remove such blocks.
        
        current_indices = list(selected_block_indices)
        changed = True
        while changed:
            changed = False
            # Calculate coverage counts for the current set of blocks
            counts = np.zeros(n_triples, dtype=np.int32)
            for idx in current_indices:
                counts[block_triple_indices[idx]] += 1
                
            new_indices = []
            for idx in current_indices:
                block_triples = block_triple_indices[idx]
                # If any triple in this block is covered ONLY by this block (count == 1),
                # the block is essential.
                if np.any(counts[block_triples] == 1):
                    new_indices.append(idx)
                else:
                    # Block is redundant. Remove it.
                    changed = True
                    # Decrement counts for the triples in this block
                    counts[block_triples] -= 1
            
            current_indices = new_indices
            
        # Return the actual blocks
        return [list(all_blocks[idx]) for idx in current_indices]

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    n_restarts = 100
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
