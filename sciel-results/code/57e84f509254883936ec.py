import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(all_t_subsets)
    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(all_k_blocks)
    
    # Bitsets for each block
    block_masks = [0] * num_k
    for j, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            mask |= (1 << idx)
        block_masks[j] = mask
        
    all_mask = (1 << num_t) - 1
    
    # Precompute ts_to_blocks
    ts_to_blocks = [[] for _ in range(num_t)]
    for j, block in enumerate(all_k_blocks):
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            ts_to_blocks[idx].append(j)

    def polish(block_indices):
        """Remove redundant blocks. Returns new list and count removed."""
        if not block_indices:
            return block_indices, 0
            
        covered = 0
        for b in block_indices:
            covered |= block_masks[b]
            
        new_indices = []
        removed = 0
        for b in block_indices:
            if time.time() > deadline - 5.0:
                break
            if (covered & ~block_masks[b]) == covered:
                removed += 1
            else:
                new_indices.append(b)
                
        return new_indices, removed

    def fast_greedy_v2(seed=None):
        if seed is not None:
            random.seed(seed)
            
        uncovered = all_mask
        selected = []
        selected_set = set()
        
        # gain[b] = number of uncovered subsets covered by b
        gain = [0] * num_k
        for b in range(num_k):
            gain[b] = block_masks[b].bit_count()
            
        while uncovered:
            if time.time() > deadline - 5.0:
                return None
            
            max_gain = -1
            best_b = -1
            for b in range(num_k):
                if b in selected_set:
                    continue
                if gain[b] > max_gain:
                    max_gain = gain[b]
                    best_b = b
                    
            if best_b == -1:
                break
                
            selected.append(best_b)
            selected_set.add(best_b)
            
            newly_covered = uncovered & block_masks[best_b]
            uncovered &= ~block_masks[best_b]
            
            u = newly_covered
            while u:
                lsb = u & -u
                ts_idx = lsb.bit_length() - 1
                u ^= lsb
                
                for c in ts_to_blocks[ts_idx]:
                    if c not in selected_set:
                        gain[c] -= 1
                        
        return selected

    def local_search_1(block_indices):
        if not block_indices:
            return block_indices
            
        covered = 0
        for b in block_indices:
            covered |= block_masks[b]
            
        n = len(block_indices)
        selected_set = set(block_indices)
        
        # Try to find a swap that allows removal
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
                
            b = block_indices[i]
            covered_without_b = covered & ~block_masks[b]
            unique_b = block_masks[b] & ~covered_without_b
            
            if unique_b == 0:
                continue
                
            unique_indices = []
            u = unique_b
            while u:
                lsb = u & -u
                unique_indices.append(lsb.bit_length() - 1)
                u ^= lsb
                
            if not unique_indices:
                continue
                
            # Try a few random candidates
            for _ in range(20):
                if time.time() > deadline - 1.0:
                    break
                    
                ts_idx = unique_indices[random.randint(0, len(unique_indices)-1)]
                if not ts_to_blocks[ts_idx]:
                    continue
                c = ts_to_blocks[ts_idx][random.randint(0, len(ts_to_blocks[ts_idx])-1)]
                
                if (block_masks[c] & unique_b) == unique_b:
                    # Valid candidate
                    block_indices[i] = c
                    selected_set.remove(b)
                    selected_set.add(c)
                    
                    new_indices, removed = polish(block_indices)
                    if removed > 0:
                        return new_indices
                    else:
                        # Revert
                        block_indices[i] = b
                        selected_set.remove(c)
                        selected_set.add(b)
                        
        return block_indices

    best_blocks = None
    best_len = float('inf')
    
    # Initial Greedy
    sol = fast_greedy_v2()
    if sol:
        sol, _ = polish(sol)
        sol = local_search_1(sol)
        sol, _ = polish(sol)
        best_blocks = sol[:]
        best_len = len(sol)
        
    # Random Restarts
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        sol = fast_greedy_v2(seed=i)
        if not sol:
            continue
            
        sol, _ = polish(sol)
        
        # Local Search
        for _ in range(2):
            if time.time() > deadline - 5.0:
                break
            sol = local_search_1(sol)
            sol, _ = polish(sol)
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_blocks = sol[:]
            
    # Final Polish
    if best_blocks:
        best_blocks, _ = polish(best_blocks)
        
    if best_blocks is None:
        return []
        
    result = [list(all_k_blocks[b]) for b in best_blocks]
    return result
