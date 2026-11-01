import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0 # Safety margin

    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        return int(np.prod([n-i for i in range(r)]) / np.prod([i+1 for i in range(r)]))
        
    t_count = nCr(k, t)
    
    # block_t_indices[i] is an array of indices into all_ts for the i-th block
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = list(itertools.combinations(block, t))
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    # Build reverse map: triple -> list of blocks covering it
    # This is a list of lists
    triple_to_blocks = [[] for _ in range(n_ts)]
    for i in range(n_ks):
        for triple_idx in block_t_indices[i]:
            triple_to_blocks[triple_idx].append(i)
            
    # Convert to numpy arrays for faster access? 
    # List of lists is fine if we just iterate.
    
    def greedy_cover(seed):
        np.random.seed(seed)
        covered = np.zeros(n_ts, dtype=bool)
        potential_gain = np.full(n_ks, t_count, dtype=np.int32)
        blocks_indices = []
        
        while time.time() - start_time < time_limit * 0.4:
            # Find block with max potential_gain
            # If max is 0, break
            max_gain = np.max(potential_gain)
            if max_gain == 0:
                break
            
            # Random tie-breaking
            candidates = np.where(potential_gain == max_gain)[0]
            best_idx = candidates[np.random.randint(len(candidates))]
            
            blocks_indices.append(best_idx)
            
            # Update covered and potential_gain
            for triple_idx in block_t_indices[best_idx]:
                if not covered[triple_idx]:
                    covered[triple_idx] = True
                    # Decrement gain for all blocks covering this triple
                    for b_prime in triple_to_blocks[triple_idx]:
                        potential_gain[b_prime] -= 1
            
            if np.all(covered):
                break
                
        return blocks_indices

    def local_search(blocks_indices):
        current_blocks = list(blocks_indices)
        
        # Helper: remove redundant blocks
        def remove_redundant(blocks):
            if not blocks:
                return []
            
            changed = True
            while changed:
                changed = False
                new_blocks = []
                
                cov_count = np.zeros(n_ts, dtype=np.int32)
                for b in blocks:
                    cov_count[block_t_indices[b]] += 1
                    
                for b in blocks:
                    min_cov = np.min(cov_count[block_t_indices[b]])
                    if min_cov > 1:
                        changed = True
                    else:
                        new_blocks.append(b)
                        
                blocks = new_blocks
            return blocks

        current_blocks = remove_redundant(current_blocks)
        
        last_improvement = time.time()
        no_improve_steps = 0
        max_no_improve = 1000
        
        while time.time() - start_time < time_limit * 0.9:
            if time.time() - last_improvement > 10.0:
                 break
            
            rand_idx = np.random.randint(n_ks)
            
            # Check if already in set (linear scan is ok for ~70 items)
            if rand_idx in current_blocks:
                continue
                
            new_blocks = current_blocks + [rand_idx]
            optimized_blocks = remove_redundant(new_blocks)
            
            if len(optimized_blocks) < len(current_blocks):
                current_blocks = optimized_blocks
                last_improvement = time.time()
                no_improve_steps = 0
            elif len(optimized_blocks) == len(current_blocks):
                if np.random.rand() < 0.05:
                    current_blocks = optimized_blocks
                no_improve_steps += 1
            else:
                no_improve_steps += 1
                
            if no_improve_steps > max_no_improve:
                break
                
        return current_blocks

    best_blocks = None
    best_len = float('inf')
    
    # Phase 1: Multiple Greedy Restarts
    num_restarts = 30 # Increased due to faster greedy
    
    for i in range(num_restarts):
        if time.time() - start_time > time_limit * 0.5:
            break
            
        blocks = greedy_cover(i)
        blocks = local_search(blocks)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
    # Phase 2: Deep Local Search on Best
    if best_blocks:
        best_blocks = local_search(best_blocks)
        
    final_blocks = [list(all_ks[i]) for i in best_blocks]
    
    return final_blocks
