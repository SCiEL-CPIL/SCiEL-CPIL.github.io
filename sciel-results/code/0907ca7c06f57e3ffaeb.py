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
            
    def greedy_with_prune(random_seed):
        rng = np.random.RandomState(random_seed)
        
        # coverage_count[i] is the number of selected blocks covering triple i
        coverage_count = np.zeros(n_triples, dtype=np.int32)
        selected_indices = []
        
        while np.any(coverage_count == 0):
            # Calculate gains: number of *uncovered* (count==0) triples in each block
            # We need to know which triples are uncovered.
            # Instead of a boolean mask, we can just check coverage_count == 0
            # But np.sum on a condition is fast.
            # gains = number of triples in block where coverage_count is 0
            
            # Efficiently calculate gains:
            # For each block, count how many of its triples have coverage_count == 0
            # This requires checking coverage_count[block_triple_indices] == 0
            
            # This creates a (n_blocks, k_t) boolean array, which is ~346k * 35 ~ 12M bools.
            # This is acceptable in memory and speed for a few iterations.
            
            uncovered_flags = (coverage_count == 0)
            # gains = sum of uncovered_flags over each block's triples
            gains = np.sum(uncovered_flags[block_triple_indices], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
                
            candidates = np.where(gains == max_gain)[0]
            best_idx = candidates[rng.randint(len(candidates))]
            
            # Update state
            selected_indices.append(best_idx)
            block_triples = block_triple_indices[best_idx]
            coverage_count[block_triples] += 1
            
        # Pruning Phase
        # We have a list of selected block indices.
        # We want to remove any block that is redundant.
        # A block is redundant if removing it doesn't drop any coverage_count to 0.
        
        # We can iterate multiple times to remove blocks that become redundant after others are removed.
        # Or just one pass in random order? Multiple passes is safer.
        
        # Create a set of active blocks for O(1) lookup? Or just a list and filter?
        # Since n_blocks is large, but selected is small (~76), a list is fine.
        
        active_blocks = list(selected_indices)
        
        # To make pruning faster and effective, we can shuffle and try to remove.
        # We need to maintain coverage_count correctly.
        
        # Current coverage_count reflects all active_blocks.
        
        changed = True
        passes = 0
        max_passes = 10 # Safety limit
        
        while changed and passes < max_passes:
            changed = False
            passes += 1
            # Shuffle active blocks to randomize removal order
            rng.shuffle(active_blocks)
            
            new_active_blocks = []
            for b_idx in active_blocks:
                block_triples = block_triple_indices[b_idx]
                
                # Check if any triple in this block would become uncovered if we remove it
                # i.e., is coverage_count[t] == 1 for any t in block_triples?
                # If yes, removing b_idx makes count 0 -> Invalid.
                # If no, removing b_idx keeps count >= 1 -> Valid.
                
                # We need to check min(coverage_count[block_triples]) > 1
                # Or simply: np.any(coverage_count[block_triples] == 1)
                
                is_redundant = not np.any(coverage_count[block_triples] == 1)
                
                if is_redundant:
                    # Remove it
                    coverage_count[block_triples] -= 1
                    changed = True
                else:
                    # Keep it
                    new_active_blocks.append(b_idx)
            
            active_blocks = new_active_blocks
            
        # Convert indices to blocks
        return [list(all_blocks[i]) for i in active_blocks]

    # Run multiple times with different seeds
    best_blocks = None
    best_len = float('inf')
    
    # Pruning adds overhead, so fewer restarts might be better, or same.
    # Let's try 10 restarts.
    n_restarts = 10
    
    for i in range(n_restarts):
        blocks = greedy_with_prune(i)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    return best_blocks
