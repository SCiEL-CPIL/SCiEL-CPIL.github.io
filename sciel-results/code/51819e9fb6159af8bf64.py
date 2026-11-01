import numpy as np
import itertools
import time
import random
from collections import Counter

def construct(v, k, t):
    start_time = time.time()
    # Hard wall-clock limit with safety margin. The evaluator kills at 150s;
    # we target returning by 115s to be safe.
    time_limit = 115.0
    
    # Precomputation
    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Generate all k-subsets
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    def nCr(n, r):
        if r > n: return 0
        if r == 0: return 1
        res = 1
        for i in range(r):
            res *= (n - i)
            res //= (i + 1)
        return res
            
    t_count = nCr(k, t)
    
    # Precompute block_t_indices
    block_t_indices = np.zeros((n_ks, t_count), dtype=np.int32)
    for i, block in enumerate(all_ks):
        subs = itertools.combinations(block, t)
        for j, sub in enumerate(subs):
            block_t_indices[i, j] = idx_map[sub]
            
    best_solution = None
    best_score = float('inf')
    
    def run_greedy(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        while remaining > 0:
            if time.time() - start_time > time_limit - 5.0:
                return None, None
            
            covered_counts = covered[block_t_indices].sum(axis=1)
            gains = t_count - covered_counts
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            if n_ks > 20:
                top_indices = np.argpartition(gains, -20)[-20:]
            else:
                top_indices = np.arange(n_ks)
                
            best_idx = top_indices[random.randint(0, len(top_indices) - 1)]
            
            blocks_indices.append(best_idx)
            
            subs_to_cover = block_t_indices[best_idx]
            prev_state = covered[subs_to_cover]
            new_covers = np.sum(~prev_state)
            remaining -= new_covers
            covered[subs_to_cover] = True
            
        current_blocks = list(blocks_indices)
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 2.0:
                return current_blocks, len(current_blocks)
                
            improved = False
            new_blocks = []
            
            coverage_count = np.zeros(n_ts, dtype=np.int32)
            for b_idx in current_blocks:
                coverage_count[block_t_indices[b_idx]] += 1
                
            for b_idx in current_blocks:
                subs_in_block = block_t_indices[b_idx]
                if np.min(coverage_count[subs_in_block]) > 1:
                    improved = True
                    coverage_count[subs_in_block] -= 1
                else:
                    new_blocks.append(b_idx)
                    
            current_blocks = new_blocks
            
        return current_blocks, len(current_blocks)

    restart_count = 0
    while time.time() - start_time < time_limit - 10.0:
        seed = restart_count
        blocks, score = run_greedy(seed)
        
        if blocks is None:
            break
            
        if score < best_score:
            best_score = score
            best_solution = blocks
            
        restart_count += 1
        
    if best_solution is None:
        blocks, score = run_greedy(0)
        if blocks is not None:
            best_solution = blocks
            best_score = score
        else:
            return []

    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks
