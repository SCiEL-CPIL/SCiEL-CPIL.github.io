import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    timeout = 115.0
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    t_subset_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    
    block_t_indices = []
    for block in all_blocks:
        indices = [t_subset_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_t_indices.append(np.array(indices, dtype=np.int32))
    
    best_blocks = None
    best_count = float('inf')
    
    def is_valid(used_blocks):
        if not used_blocks:
            return len(uncovered) == 0 if 'uncovered' in locals() else False
        cov = np.zeros(num_t_subsets, dtype=np.int8)
        for b_idx in used_blocks:
            cov[block_t_indices[b_idx]] += 1
        return np.all(cov >= 1)
    
    def delete_redundant(used_blocks, max_passes=5):
        if not used_blocks:
            return used_blocks
        coverage_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in used_blocks:
            coverage_counts[block_t_indices[b_idx]] += 1
        
        changed = True
        passes = 0
        while changed and passes < max_passes:
            changed = False
            passes += 1
            new_used = []
            for b_idx in used_blocks:
                t_idx = block_t_indices[b_idx]
                if np.all(coverage_counts[t_idx] >= 2):
                    coverage_counts[t_idx] -= 1
                    changed = True
                else:
                    new_used.append(b_idx)
            used_blocks = new_used
        return used_blocks
    
    def local_search(used_blocks, time_limit):
        if not used_blocks:
            return used_blocks
        coverage_counts = np.zeros(num_t_subsets, dtype=np.int32)
        for b_idx in used_blocks:
            coverage_counts[block_t_indices[b_idx]] += 1
        
        current_size = len(used_blocks)
        used_set = set(used_blocks)
        
        while time.time() - start_time < time_limit:
            improved = False
            for i, b_idx in enumerate(used_blocks):
                if time.time() - start_time > time_limit:
                    break
                if b_idx not in used_set:
                    continue
                    
                coverage_counts[block_t_indices[b_idx]] -= 1
                remaining = set(used_set) - {b_idx}
                
                if np.all(coverage_counts >= 1):
                    used_blocks[i] = None
                    used_set.discard(b_idx)
                    improved = True
                    continue
                
                coverage_counts[block_t_indices[b_idx]] += 1
                
                uncovered_by_b = block_t_indices[b_idx][coverage_counts < 2]
                if len(uncovered_by_b) == 0:
                    continue
                    
                best_swap = None
                best_swap_size = current_size + 1
                
                for j, c_idx in enumerate(used_blocks):
                    if c_idx is None or c_idx == b_idx or c_idx in used_set:
                        continue
                    if c_idx in used_set:
                        continue
                    t_idx_c = block_t_indices[c_idx]
                    covers_uncovered = np.isin(uncovered_by_b, t_idx_c)
                    if covers_uncovered.all():
                        if len(used_set) + 1 < best_swap_size:
                            best_swap = c_idx
                            best_swap_size = len(used_set) + 1
                if best_swap is not None:
                    coverage_counts[block_t_indices[b_idx]] -= 1
                    coverage_counts[block_t_indices[best_swap]] += 1
                    used_blocks[i] = best_swap
                    used_set.discard(b_idx)
                    used_set.add(best_swap)
                    current_size = len(used_set)
                    improved = True
        
        return [b for b in used_blocks if b is not None]
    
    while time.time() - start_time < timeout:
        uncovered = np.ones(num_t_subsets, dtype=bool)
        used_blocks = []
        
        while uncovered.any():
            best_gain = -1
            best_block_idx = -1
            candidates = []
            
            for i in range(num_blocks):
                if len(block_t_indices[i]) <= best_gain:
                    continue
                gain = np.count_nonzero(block_t_indices[i][uncovered])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain and gain > 0:
                    candidates.append(i)
            
            if best_gain == 0:
                break
                
            selected_idx = random.choice(candidates)
            used_blocks.append(selected_idx)
            uncovered[block_t_indices[selected_idx]] = False
            
            if time.time() - start_time > timeout - 5:
                break
                
        if time.time() - start_time > timeout - 5:
            break
            
        used_blocks = delete_redundant(used_blocks, max_passes=3)
        
        if time.time() - start_time < timeout - 10:
            used_blocks = local_search(used_blocks, timeout - 10)
            used_blocks = delete_redundant(used_blocks, max_passes=2)
        
        if len(used_blocks) < best_count:
            best_count = len(used_blocks)
            best_blocks = [all_blocks[i] for i in used_blocks]
            
    if best_blocks is None:
        uncovered = np.ones(num_t_subsets, dtype=bool)
        used_blocks = []
        while uncovered.any():
            best_gain = -1
            best_block_idx = -1
            for i in range(num_blocks):
                gain = np.count_nonzero(block_t_indices[i][uncovered])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            used_blocks.append(best_block_idx)
            uncovered[block_t_indices[best_block_idx]] = False
        best_blocks = [all_blocks[i] for i in used_blocks]
        
    return best_blocks
