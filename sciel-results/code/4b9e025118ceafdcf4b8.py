import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # masks[i] is an integer bitmask of t-subsets covered by k-subset i
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
        return x.bit_count()

    # Greedy construction
    def greedy_construct(uncovered_mask, randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
            
        current_blocks = []
        used_blocks = set()
        uncovered = uncovered_mask
        
        # gain[i] = number of uncovered t-subsets covered by i
        gain = [0] * num_k_subsets
        for i in range(num_k_subsets):
            gain[i] = popcount(masks[i] & uncovered)
            
        while uncovered:
            if time.time() > deadline:
                return None
                
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
                return None # Should not happen
                
            if randomize and candidates:
                chosen_idx = random.choice(candidates)
            else:
                chosen_idx = candidates[0] if candidates else -1
                
            current_blocks.append(chosen_idx)
            used_blocks.add(chosen_idx)
            
            # Update uncovered and gains
            newly_covered = uncovered & masks[chosen_idx]
            uncovered &= ~masks[chosen_idx]
            
            # Update gains for blocks covering newly covered t-subsets
            nc = newly_covered
            while nc:
                lsb = nc & -nc
                idx_j = lsb.bit_length() - 1
                nc ^= lsb
                for i in t_to_k[idx_j]:
                    if i not in used_blocks:
                        gain[i] -= 1
                        if gain[i] < 0: gain[i] = 0
                        
        return current_blocks

    def local_search(block_indices):
        if not block_indices:
            return []
            
        current_blocks = list(block_indices)
        selected = set(current_blocks)
        
        # cover_counts[j] = number of selected blocks covering t-subset j
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
                
        # unique_counts[i] = number of t-subsets covered only by block i (among selected)
        unique_counts = [0] * num_k_subsets
        
        # Initialize unique_counts
        for j in range(num_t_subsets):
            if cover_counts[j] == 1:
                # Find the unique block
                for i in t_to_k[j]:
                    if i in selected:
                        unique_counts[i] += 1
                        break
                        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find all blocks with unique_counts == 0
            deletable = [i for i in current_blocks if unique_counts[i] == 0]
            
            if not deletable:
                break
                
            # Randomly pick one to delete
            idx_to_remove = random.choice(deletable)
            
            # Remove it
            current_blocks.remove(idx_to_remove)
            selected.remove(idx_to_remove)
            
            # Update cover_counts and unique_counts
            mask = masks[idx_to_remove]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                m ^= lsb
                
                old_count = cover_counts[j]
                cover_counts[j] = old_count - 1
                
                if old_count == 2:
                    # Now count is 1. Find the new unique block.
                    for i in t_to_k[j]:
                        if i in selected:
                            unique_counts[i] += 1
                            break
                            
            improved = True
            
        return current_blocks

    def perturb_and_repair(block_indices, num_perturb):
        current_blocks = list(block_indices)
        selected = set(current_blocks)
        
        # Randomly remove some blocks
        to_remove = random.sample(current_blocks, min(num_perturb, len(current_blocks)))
        for idx in to_remove:
            current_blocks.remove(idx)
            selected.remove(idx)
            
        # Calculate uncovered
        cover_counts = [0] * num_t_subsets
        for idx in current_blocks:
            mask = masks[idx]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                cover_counts[j] += 1
                m ^= lsb
                
        uncovered_mask = 0
        for j in range(num_t_subsets):
            if cover_counts[j] == 0:
                uncovered_mask |= (1 << j)
                
        if uncovered_mask == 0:
            return current_blocks
            
        # Greedy repair
        added = greedy_construct(uncovered_mask, randomize=True)
        if added is None:
            return None
            
        # Combine
        for idx in added:
            if idx not in selected:
                current_blocks.append(idx)
                selected.add(idx)
                
        return current_blocks

    best_blocks = None
    best_count = float('inf')
    
    # Initial Greedy
    init_blocks = greedy_construct((1 << num_t_subsets) - 1, randomize=False)
    if init_blocks:
        init_blocks = local_search(init_blocks)
        best_blocks = init_blocks
        best_count = len(init_blocks)
        
    # ILS Loop
    while time.time() < deadline:
        # Perturb best solution
        # Use variable perturbation size
        num_perturb = random.randint(3, 20)
        perturbed = perturb_and_repair(best_blocks, num_perturb)
        
        if perturbed is None:
            continue
            
        # Local Search
        improved = local_search(perturbed)
        
        if len(improved) < best_count:
            best_count = len(improved)
            best_blocks = improved
            
    if best_blocks is None:
        # Fallback
        return []
        
    return [list(k_subsets[i]) for i in best_blocks]
