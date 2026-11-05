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
    
    # Precompute block_triples: list of lists of triple indices for each block
    block_triples = []
    # Precompute triple_to_blocks: list of lists of block indices for each triple
    triple_to_blocks = [[] for _ in range(n_triples)]
    
    for i, block in enumerate(all_blocks):
        b_triples = []
        for sub in itertools.combinations(block, t):
            t_idx = triple_to_idx[sub]
            b_triples.append(t_idx)
            triple_to_blocks[t_idx].append(i)
        block_triples.append(b_triples)
        
    def greedy_cover(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # Initial gain for each block is k_t (all its triples are uncovered)
        gains = np.full(n_blocks, k_t, dtype=np.int32)
        uncovered_mask = np.ones(n_triples, dtype=bool)
        selected_mask = np.zeros(n_blocks, dtype=bool)
        
        selected_blocks = []
        remaining = n_triples
        
        while remaining > 0:
            # Find max gain among unselected blocks
            # Mask out selected blocks by setting gain to -1? 
            # Or just filter. Filtering creates copies. 
            # Setting to -1 is in-place and fast.
            gains[selected_mask] = -1
            
            max_gain = np.max(gains)
            if max_gain <= 0:
                break
                
            # Find candidates with max gain
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Select block
            selected_mask[best_idx] = True
            selected_blocks.append(list(all_blocks[best_idx]))
            
            # Update gains for affected blocks
            # Only triples that were actually newly covered affect other blocks' gains
            # If a triple was already covered, it didn't contribute to this block's gain,
            # so it shouldn't reduce other blocks' gains? 
            # Wait. 
            # Gain of block B' = number of uncovered triples in B'.
            # If we cover triple T, and T is in B', then gain of B' decreases by 1.
            # This is true regardless of whether T was covered by *this* block previously?
            # No, T is covered *now* by this block. Before, it was uncovered.
            # So yes, for every triple T in the selected block that was *previously uncovered*,
            # we must decrement the gain of every other block containing T.
            
            newly_covered = []
            for t_idx in block_triples[best_idx]:
                if uncovered_mask[t_idx]:
                    uncovered_mask[t_idx] = False
                    newly_covered.append(t_idx)
                    remaining -= 1
            
            if newly_covered:
                # Vectorized update is hard with list of lists.
                # Loop over newly covered triples and update gains of their blocks.
                for t_idx in newly_covered:
                    affected = triple_to_blocks[t_idx]
                    if affected:
                        gains[affected] -= 1
                        
        return selected_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Increase restarts due to efficiency
    n_restarts = 50
    
    for i in range(n_restarts):
        blocks = greedy_cover(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
