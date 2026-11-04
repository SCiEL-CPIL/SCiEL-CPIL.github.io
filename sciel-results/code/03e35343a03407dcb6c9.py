import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    points = range(v)
    all_subsets = list(itertools.combinations(points, t))
    num_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # Precompute masks for each block
    # mask[i] is an integer where bit j is 1 if block i covers subset j
    masks = [0] * num_blocks
    # Also store subset indices for greedy gain calculation
    block_to_subsets = [[] for _ in range(num_blocks)]
    
    for b_idx, block in enumerate(all_blocks):
        mask_val = 0
        subs = []
        for s in itertools.combinations(block, t):
            s_idx = subset_to_idx[s]
            mask_val |= (1 << s_idx)
            subs.append(s_idx)
        masks[b_idx] = mask_val
        block_to_subsets[b_idx] = subs
        
    ALL_ONES = (1 << num_subsets) - 1
    
    def run_greedy(random_seed=None):
        rng = np.random.default_rng(random_seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            if random_seed is not None:
                noise = rng.uniform(0, 0.1, size=num_blocks)
                current_gains = gains + noise
            else:
                current_gains = gains
                
            best_idx = np.argmax(current_gains)
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                continue
                
            for s in newly_covered:
                uncovered[s] = False
                
            # Update gains
            # We need to decrement gains for all blocks that cover any newly covered subset
            # To do this efficiently, we can precompute subset_to_blocks
            # But we didn't in this scope. Let's do it on the fly or precompute globally.
            # Precomputing subset_to_blocks is better.
            
        return selected_blocks

    # Precompute subset_to_blocks for faster gain updates
    subset_to_blocks = [[] for _ in range(num_subsets)]
    for b_idx in range(num_blocks):
        for s_idx in block_to_subsets[b_idx]:
            subset_to_blocks[s_idx].append(b_idx)

    # Re-define run_greedy to use subset_to_blocks
    def run_greedy_optimized(random_seed=None):
        rng = np.random.default_rng(random_seed)
        uncovered = np.ones(num_subsets, dtype=bool)
        gains = np.array([len(bs) for bs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while True:
            if random_seed is not None:
                noise = rng.uniform(0, 0.1, size=num_blocks)
                current_gains = gains + noise
            else:
                current_gains = gains
                
            best_idx = np.argmax(current_gains)
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            b_subs = block_to_subsets[best_idx]
            newly_covered = [s for s in b_subs if uncovered[s]]
            
            if not newly_covered:
                continue
                
            for s in newly_covered:
                uncovered[s] = False
                
            # Update gains
            # Use a set to avoid double counting if a block covers multiple newly covered subsets?
            # No, gains[b] is the number of *uncovered* subsets covered by b.
            # If b covers s1 and s2, and both become covered, gains[b] should decrease by 2.
            # So we just iterate through newly_covered and decrement gains for all blocks covering them.
            # If a block b covers s1 and s2, it will be encountered twice.
            # We must ensure we don't double decrement if we aren't careful?
            # Actually, gains[b] counts distinct subsets.
            # If s1 and s2 are distinct, decrementing twice is correct.
            
            for s in newly_covered:
                for b in subset_to_blocks[s]:
                    gains[b] -= 1
                    
        return selected_blocks

    def minimize(blocks_indices):
        current_blocks = list(blocks_indices)
        # Randomized minimization
        improved = True
        while improved:
            improved = False
            random.shuffle(current_blocks)
            for i in range(len(current_blocks)):
                # Check if current_blocks[i] is redundant
                # Compute OR of all other masks
                or_val = 0
                for j in range(len(current_blocks)):
                    if j != i:
                        or_val |= masks[current_blocks[j]]
                
                if or_val == ALL_ONES:
                    current_blocks.pop(i)
                    improved = True
                    break # Restart loop
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple trials
    # The greedy part is O(N_blocks * N_subsets) roughly.
    # Minimization is O(N_final^2 * N_final) worst case per pass, but usually fast.
    # With 120s, we can afford ~20-50 trials.
    
    start_time = time.time()
    trial = 0
    while time.time() - start_time < 110: # Leave 10s buffer
        seed = trial
        greedy_res = run_greedy_optimized(seed)
        min_res = minimize(greedy_res)
        
        if len(min_res) < best_len:
            best_len = len(min_res)
            best_blocks = min_res
            # print(f"Trial {seed}: {len(min_res)} blocks") # Debug
            
        trial += 1
        
    if best_blocks is None:
        # Fallback
        return [all_blocks[i] for i in run_greedy_optimized(0)]
        
    return [all_blocks[i] for i in best_blocks]
