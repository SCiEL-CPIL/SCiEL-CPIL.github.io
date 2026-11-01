import itertools
import random
import time

def construct(v, k, t):
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    t_subset_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_k_blocks)
    
    # Precompute t-subsets for each block as a set of indices
    block_t_indices = []
    for block in all_k_blocks:
        t_indices = set(t_subset_index[ts] for ts in itertools.combinations(block, t))
        block_t_indices.append(t_indices)
        
    # Precompute ts to block indices for fast candidate lookup
    ts_to_block_indices = [[] for _ in range(num_t_subsets)]
    for b_idx, t_indices in enumerate(block_t_indices):
        for t_idx in t_indices:
            ts_to_block_indices[t_idx].append(b_idx)

    # 1. Greedy Initialization
    uncovered = set(range(num_t_subsets))
    current_block_indices = []
    
    while uncovered:
        target_idx = random.choice(list(uncovered))
        
        best_block_idx = -1
        best_gain = -1
        
        # Only check blocks that contain target_idx
        for b_idx in ts_to_block_indices[target_idx]:
            gain = 0
            for t_idx in block_t_indices[b_idx]:
                if t_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        current_block_indices.append(best_block_idx)
        for t_idx in block_t_indices[best_block_idx]:
            uncovered.discard(t_idx)
            
    # 2. Local Search
    counts = [0] * num_t_subsets
    for b_idx in current_block_indices:
        for t_idx in block_t_indices[b_idx]:
            counts[t_idx] += 1
            
    improved = True
    max_iterations = 100
    iter_count = 0
    start_time = time.time()
    
    while improved and iter_count < max_iterations and (time.time() - start_time) < 100:
        improved = False
        iter_count += 1
        
        # 1. Remove redundant blocks
        i = 0
        while i < len(current_block_indices):
            b_idx = current_block_indices[i]
            redundant = True
            for t_idx in block_t_indices[b_idx]:
                if counts[t_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                for t_idx in block_t_indices[b_idx]:
                    counts[t_idx] -= 1
                current_block_indices.pop(i)
                improved = True
            else:
                i += 1
                
        # 2. Swap
        if not improved:
            swapped = False
            for i in range(len(current_block_indices)):
                b_old_idx = current_block_indices[i]
                critical_ts = [t_idx for t_idx in block_t_indices[b_old_idx] if counts[t_idx] == 1]
                
                if not critical_ts:
                    continue
                    
                # Get candidates from the first critical ts
                candidates = ts_to_block_indices[critical_ts[0]]
                # Shuffle to avoid bias
                random.shuffle(candidates)
                
                for b_new_idx in candidates:
                    if b_new_idx == b_old_idx:
                        continue
                    
                    # Check if b_new covers all critical_ts
                    covers_all = True
                    t_set_new = block_t_indices[b_new_idx]
                    for t_idx in critical_ts:
                        if t_idx not in t_set_new:
                            covers_all = False
                            break
                            
                    if covers_all:
                        # Swap
                        for t_idx in block_t_indices[b_old_idx]:
                            counts[t_idx] -= 1
                        for t_idx in t_set_new:
                            counts[t_idx] += 1
                            
                        current_block_indices[i] = b_new_idx
                        improved = True
                        swapped = True
                        break
                        
                if swapped:
                    break
                    
    return [all_k_blocks[i] for i in current_block_indices]
