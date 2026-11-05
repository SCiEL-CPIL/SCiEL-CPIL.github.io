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
        # Use a local RandomState to avoid global side effects
        rng = np.random.RandomState(random_seed)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_block_indices = []
        remaining = n_triples
        
        while remaining > 0:
            # Calculate gains: number of uncovered triples in each block
            # Directive: maximize ratio of newly covered triplets to points in block (k).
            # Since k is constant, this is equivalent to maximizing newly covered triplets.
            gains = np.sum(uncovered_mask[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one candidate to break ties randomly
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            block_idx = block_triple_indices[best_idx]
            uncovered_mask[block_idx] = False
            remaining -= max_gain
            selected_block_indices.append(best_idx)
            
        return selected_block_indices

    def minimize_covering(block_indices):
        """
        Removes redundant blocks from the covering.
        A block is redundant if all its triples are covered by other blocks in the set.
        """
        if not block_indices:
            return []
            
        cover_count = np.zeros(n_triples, dtype=np.int32)
        
        # Initialize coverage counts
        for b_idx in block_indices:
            triples = block_triple_indices[b_idx]
            cover_count[triples] += 1
            
        current_indices = list(block_indices)
        changed = True
        
        while changed:
            changed = False
            new_indices = []
            for b_idx in current_indices:
                triples = block_triple_indices[b_idx]
                # If the minimum coverage count among this block's triples is >= 2,
                # then every triple is covered by at least one other block.
                min_count = np.min(cover_count[triples])
                
                if min_count >= 2:
                    # Block is redundant, remove it
                    cover_count[triples] -= 1
                    changed = True
                else:
                    # Block is essential, keep it
                    new_indices.append(b_idx)
                    
            current_indices = new_indices
            
        return current_indices

    # Run multiple times with different seeds to find a better covering
    best_indices = None
    best_len = float('inf')
    
    # Increased restarts to utilize the 120s time limit
    n_restarts = 80
    
    for i in range(n_restarts):
        indices = greedy_cover(i)
        minimal_indices = minimize_covering(indices)
        
        if len(minimal_indices) < best_len:
            best_len = len(minimal_indices)
            best_indices = minimal_indices
            
    # Convert indices back to lists of points
    return [list(all_blocks[i]) for i in best_indices]
