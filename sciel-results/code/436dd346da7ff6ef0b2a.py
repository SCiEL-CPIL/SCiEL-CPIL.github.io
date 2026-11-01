import time
import itertools
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {s: i for i, s in enumerate(t_subsets)}
    M = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    N = len(k_subsets)
    
    block_indices = []
    block_masks = []
    for block in k_subsets:
        idxs = []
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_sub_to_idx[t_sub]
            idxs.append(idx)
            mask |= (1 << idx)
        block_indices.append(idxs)
        block_masks.append(mask)
        
    full_mask = (1 << M) - 1
    
    def greedy_eliminate(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        uncovered = full_mask
        selected_indices = []
        
        while uncovered:
            if time.time() > deadline - safety_margin:
                return None # Timeout
            
            best_gain = 0
            best_candidates = []
            
            # Find all blocks with max gain
            for i in range(N):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
            
            if best_gain == 0:
                return None
                
            # Pick one candidate
            if len(best_candidates) == 1:
                best_idx = best_candidates[0]
            else:
                # Random choice to allow different paths
                best_idx = random.choice(best_candidates)
                
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
        # Elimination
        counts = np.zeros(M, dtype=np.int32)
        for idx in selected_indices:
            idxs = block_indices[idx]
            counts[idxs] += 1
            
        changed = True
        while changed:
            if time.time() > deadline - safety_margin:
                return None
            changed = False
            for idx in list(selected_indices):
                idxs = block_indices[idx]
                if np.min(counts[idxs]) > 1:
                    counts[idxs] -= 1
                    selected_indices.remove(idx)
                    changed = True
                    break
                    
        return selected_indices

    best_result = None
    best_size = float('inf')
    
    # Run multiple times
    # First run is deterministic (or random seed 0)
    for i in range(10): # Try up to 10 times
        if time.time() > deadline - safety_margin - 10.0:
            break
            
        res = greedy_eliminate(random_seed=i)
        if res is None:
            continue
            
        size = len(res)
        if size < best_size:
            best_size = size
            best_result = res
            
        # If we found a very good result, we might stop early?
        # Theoretical lower bound is 264. If we get close, stop.
        if best_size <= 270:
            break

    if best_result is None:
        # Fallback: return empty or greedy without elimination if something went wrong
        # But greedy should always work.
        # If timeout occurred in first run, we might not have a result.
        # Let's do a quick greedy without elimination as fallback.
        uncovered = full_mask
        selected_indices = []
        while uncovered and time.time() < deadline - 1.0:
            best_gain = 0
            best_idx = -1
            for i in range(N):
                gain = (uncovered & block_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain == 0:
                break
            selected_indices.append(best_idx)
            uncovered &= ~block_masks[best_idx]
        best_result = selected_indices

    return [list(k_subsets[i]) for i in best_result]
