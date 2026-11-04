import numpy as np
import itertools
import time

def construct(v, k, t):
    # Parameters
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute coverage
    # block_to_subsets: list of lists of subset indices
    block_to_subsets = []
    # subset_to_blocks: list of lists of block indices
    subset_to_blocks = [[] for _ in range(num_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        covered_subsets = []
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            covered_subsets.append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
        block_to_subsets.append(covered_subsets)
        
    def run_greedy(random_seed=None):
        rng = np.random.default_rng(random_seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        # To avoid modifying the original gains array in a way that's hard to reset,
        # we just work on a copy.
        # The loop:
        while True:
            if random_seed is not None:
                # Add small noise to break ties randomly
                noise = rng.uniform(0, 0.1, size=num_blocks)
                current_gains = gains + noise
            else:
                current_gains = gains
                
            best_idx = np.argmax(current_gains)
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            # Determine newly covered subsets
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                # Should not happen if gains[best_idx] > 0
                continue
                
            # Mark as uncovered
            for s in newly_covered:
                uncovered[s] = False
                
            # Update gains for all blocks that cover any of the newly covered subsets
            # Use a dictionary to aggregate decrements per block
            delta_gains = {}
            for s in newly_covered:
                for b in subset_to_blocks[s]:
                    if b == best_idx:
                        continue # We don't care about updating the selected block's gain for future argmax? 
                                 # Actually, we do, to keep it 0. But we will pick it again if gain is 0? No, gain will be 0.
                                 # But we should update it to 0 to be safe.
                    delta_gains[b] = delta_gains.get(b, 0) + 1
            
            for b, d in delta_gains.items():
                gains[b] -= d
                
            # Also update the selected block's gain to 0 (or its actual remaining gain)
            # The selected block's gain should be reduced by len(newly_covered)
            gains[best_idx] -= len(newly_covered)
            
        return [all_blocks[i] for i in selected_blocks]

    # Run multiple times with different random seeds to find a better solution
    best_blocks = None
    best_len = float('inf')
    
    # One deterministic run
    blocks_det = run_greedy(random_seed=None)
    if len(blocks_det) < best_len:
        best_len = len(blocks_det)
        best_blocks = blocks_det
        
    # Several randomized runs
    for seed in range(10):
        blocks_rand = run_greedy(random_seed=seed)
        if len(blocks_rand) < best_len:
            best_len = len(blocks_rand)
            best_blocks = blocks_rand
            
    return best_blocks
