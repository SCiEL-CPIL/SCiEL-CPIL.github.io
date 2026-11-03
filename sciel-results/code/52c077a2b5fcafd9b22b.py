import random
import itertools
import numpy as np
import math
import time

def construct(v, k, t):
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subs)}
    
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # Precompute coverage lists
    coverage = []
    for k_sub in all_k_subs:
        covered_t_indices = []
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_index[t_sub]
            covered_t_indices.append(idx)
        coverage.append(covered_t_indices)
        
    # Precompute reverse index
    t_to_k = [[] for _ in range(n_t_subs)]
    for k_idx, k_sub in enumerate(all_k_subs):
        for t_sub in itertools.combinations(k_sub, t):
            t_idx = t_sub_index[t_sub]
            t_to_k[t_idx].append(k_idx)

    # State
    cover_count = np.zeros(n_t_subs, dtype=np.int32)
    in_set = np.zeros(n_k_subs, dtype=bool)
    
    # Initialization: Randomized Greedy
    # Maintain a list of uncovered indices for fast random selection
    uncovered_list = list(range(n_t_subs))
    uncovered_set = set(uncovered_list)
    
    block_indices = []
    
    while uncovered_list:
        # Pick random uncovered
        pos = random.randint(0, len(uncovered_list) - 1)
        target_t_idx = uncovered_list[pos]
        
        # Pick random k-subset covering it
        candidates = t_to_k[target_t_idx]
        k_idx = random.choice(candidates)
        
        if in_set[k_idx]:
            # If it's already in set, it shouldn't be uncovered? 
            # No, if it was in set, target_t_idx would have been covered.
            # So this shouldn't happen unless logic error.
            # But maybe another block added later covered it? 
            # No, we only add blocks. 
            # So if target is uncovered, no block in set covers it.
            # So k_idx (which covers it) cannot be in set.
            pass

        in_set[k_idx] = True
        block_indices.append(k_idx)
        
        # Update coverage
        # Find which of the covered t-indices were uncovered
        newly_covered = []
        for t_idx in coverage[k_idx]:
            if cover_count[t_idx] == 0:
                newly_covered.append(t_idx)
            cover_count[t_idx] += 1
            
        # Remove newly covered from uncovered_list
        if newly_covered:
            for t_idx in newly_covered:
                # Find index in uncovered_list
                # This is O(N). 
                # Optimization: Use a dictionary or just mark as covered and filter?
                # Given N=3876, O(N) is fast enough for initialization.
                try:
                    idx_in_list = uncovered_list.index(t_idx)
                    uncovered_list[idx_in_list] = uncovered_list[-1]
                    uncovered_list.pop()
                except ValueError:
                    pass # Already removed?

    initial_block_count = len(block_indices)
    initial_energy = initial_block_count
    
    best_energy = initial_energy
    best_in_set = in_set.copy()
    best_block_count = initial_block_count
    
    current_energy = initial_energy
    current_block_count = initial_block_count
    
    T = 5.0
    cooling_rate = 0.995
    min_T = 0.01
    
    start_time = time.time()
    time_limit = 115
    
    while T > min_T and (time.time() - start_time) < time_limit:
        move_type = random.random()
        
        if move_type < 0.6 and current_block_count > 0:
            # Remove
            # Pick random active block
            # Use np.where
            active_blocks = np.where(in_set)[0]
            if len(active_blocks) == 0:
                T *= cooling_rate
                continue
            k_idx_remove = active_blocks[random.randint(0, len(active_blocks)-1)]
            
            newly_uncovered = 0
            for t_idx in coverage[k_idx_remove]:
                if cover_count[t_idx] == 1:
                    newly_uncovered += 1
            
            delta_e = (newly_uncovered * 1000) - 1
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                in_set[k_idx_remove] = False
                for t_idx in coverage[k_idx_remove]:
                    cover_count[t_idx] -= 1
                current_block_count -= 1
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_in_set = in_set.copy()
                    best_block_count = current_block_count
                    
        else:
            # Add
            k_idx_add = random.randint(0, n_k_subs - 1)
            if in_set[k_idx_add]:
                T *= cooling_rate
                continue
                
            newly_covered = 0
            for t_idx in coverage[k_idx_add]:
                if cover_count[t_idx] == 0:
                    newly_covered += 1
            
            delta_e = -(newly_covered * 1000) + 1
            
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                in_set[k_idx_add] = True
                for t_idx in coverage[k_idx_add]:
                    cover_count[t_idx] += 1
                current_block_count += 1
                current_energy += delta_e
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_in_set = in_set.copy()
                    best_block_count = current_block_count

        T *= cooling_rate

    best_blocks = []
    for i in range(n_k_subs):
        if best_in_set[i]:
            best_blocks.append(list(all_k_subs[i]))
            
    return best_blocks
