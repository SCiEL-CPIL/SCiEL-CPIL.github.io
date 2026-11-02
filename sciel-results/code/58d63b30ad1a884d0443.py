import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask
    masks = [0] * num_k_subsets
    # t_to_k[j] is list of k-subset indices that cover t-subset j
    t_to_k = [[] for _ in range(num_t_subsets)]
    
    for i, ks in enumerate(k_subsets):
        mask = 0
        for ts in itertools.combinations(ks, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
            t_to_k[idx].append(i)
        masks[i] = mask
        
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    best_blocks = None
    best_count = float('inf')
    
    # --- Phase 1: Fast Greedy Construction with Scarcity Weighting ---
    def greedy_construct(randomize=True, seed=None):
        if seed is not None:
            random.seed(seed)
        
        uncovered = (1 << num_t_subsets) - 1
        current_blocks = []
        used_blocks = set()
        
        # gain[i] = number of uncovered t-subsets covered by k-subset i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i])
            
        # Precompute degree of each t-subset (number of blocks covering it)
        # This is static, but we can use it to weight the gain
        # degree[j] = len(t_to_k[j])
        degree = [len(lst) for lst in t_to_k]
        
        # scarcity_weight[j] = 1 / (degree[j] + 1) or similar
        # We want to prioritize covering t-subsets that are hard to cover (low degree)
        scarcity = [1.0 / (d + 1) for d in degree]
        
        # Precompute weighted_gain[i] = sum of scarcity[j] for j in mask[i]
        # This is expensive to compute from scratch each time, so we'll just use gain for now
        # and maybe adjust the selection probability based on scarcity of the newly covered sets
        
        while uncovered:
            if time.time() > deadline:
                return None, False
            
            # Find max gain
            max_gain = 0
            candidates = []
            for i in range(num_k_subsets):
                if i in used_blocks:
                    continue
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    candidates = [i]
                elif gain[i] == max_gain and gain[i] > 0:
                    candidates.append(i)
            
            if max_gain == 0:
                return current_blocks, False
            
            # Random choice among max gain candidates
            if randomize and candidates:
                # Weighted random choice based on scarcity of the sets it covers?
                # For simplicity, just uniform random among max gain
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0] if candidates else -1
            
            if chosen_idx == -1:
                return current_blocks, False
                
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Efficient gain update using t_to_k
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0

        return current_blocks, True

    def prune_redundant(block_indices):
        if not block_indices:
            return []
        
        cover_counts = [0] * num_t_subsets
        for idx in block_indices:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        changed = True
        current_blocks = list(block_indices)
        while changed:
            changed = False
            new_blocks = []
            for idx in current_blocks:
                mask = masks[idx]
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    j = lsb.bit_length() - 1
                    if cover_counts[j] == 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    m = mask
                    while m:
                        lsb = m & -m
                        j = lsb.bit_length() - 1
                        cover_counts[j] -= 1
                        m ^= lsb
                    changed = True
                else:
                    new_blocks.append(idx)
            current_blocks = new_blocks
        return current_blocks

    def local_search_improve(block_indices, max_iters=1000):
        if not block_indices:
            return []
        
        current_blocks = list(block_indices)
        current_size = len(current_blocks)
        
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
        
        def is_removable(idx):
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if cover_counts[j] == 1:
                    return False
                m ^= lsb
            return True
        
        def remove_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] -= 1
                m ^= lsb
            current_blocks.remove(idx)
            
        def add_block(idx):
            nonlocal current_blocks
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
            current_blocks.append(idx)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            random.shuffle(current_blocks)
            
            for idx in current_blocks[:]:
                if time.time() > deadline:
                    break
                if is_removable(idx):
                    remove_block(idx)
                    improved = True
                    current_size -= 1
        
        return current_blocks

    # --- Main Loop ---
    num_restarts = 1000
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        blocks, valid = greedy_construct(randomize=True)
        if not valid:
            continue
            
        pruned = prune_redundant(blocks)
        
        if len(pruned) < best_count or best_count == float('inf'):
            ls_blocks = local_search_improve(pruned)
            if len(ls_blocks) < best_count:
                best_count = len(ls_blocks)
                best_blocks = [list(k_subsets[i]) for i in ls_blocks]
        else:
            if len(pruned) < best_count:
                best_count = len(pruned)
                best_blocks = [list(k_subsets[i]) for i in pruned]

    if best_blocks is None:
        blocks, valid = greedy_construct(randomize=False)
        if valid:
            pruned = prune_redundant(blocks)
            best_blocks = [list(k_subsets[i]) for i in pruned]
        else:
            best_blocks = []

    return best_blocks
