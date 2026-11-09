import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # 1. Precompute all t-subsets and assign them indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}

    # 2. Precompute coverage bitmask for every possible k-block
    all_blocks = []
    block_masks = []
    
    for block in itertools.combinations(range(v), k):
        all_blocks.append(block)
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks.append(mask)
        
    num_blocks = len(all_blocks)
    best_solution = None
    best_size = float('inf')

    # Helper to count bits
    def count_bits(x):
        return x.bit_count()

    def greedy_cover(random_seed):
        rng = random.Random(random_seed)
        current_uncovered = (1 << num_t_subsets) - 1
        chosen_indices = []
        used = [False] * num_blocks
        
        while current_uncovered != 0:
            if time.time() > deadline:
                return None
            
            best_gain = -1
            best_block_idx = -1
            # Collect top candidates to allow random tie-breaking
            candidates = []
            
            for i in range(num_blocks):
                if used[i]:
                    continue
                intersection = block_masks[i] & current_uncovered
                gain = count_bits(intersection)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    candidates = [i]
                elif gain == best_gain:
                    candidates.append(i)
            
            if best_block_idx == -1:
                break
                # If no block covers any new subset, we are stuck (shouldn't happen with valid parameters)
                # but let's return what we have if we can't proceed
                return chosen_indices if chosen_indices else None
                
            # Randomly pick from top candidates to diversify
            chosen_idx = rng.choice(candidates)
            chosen_indices.append(chosen_idx)
            used[chosen_idx] = True
            current_uncovered &= ~block_masks[chosen_idx]
            
        return chosen_indices

    def reduce_cover(chosen_indices):
        if not chosen_indices:
            return []
            
        # Count coverage for each subset
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        
        # Helper to update counts
        def update_counts(idx, delta):
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                cover_count[idx_bit] += delta
                m ^= lsb
                
        # Initial counts
        for idx in chosen_indices:
            update_counts(idx, 1)
            
        kept_indices = []
        # Try to remove each block
        for idx in chosen_indices:
            # Check if idx is redundant
            # It is redundant if all subsets in mask[idx] have count > 1
            mask = block_masks[idx]
            m = mask
            is_redundant = True
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                if cover_count[idx_bit] <= 1:
                    is_redundant = False
                    break
                m ^= lsb
                
            if is_redundant:
                # Remove it
                update_counts(idx, -1)
            else:
                kept_indices.append(idx)
                
        return kept_indices

    def local_search_swap(chosen_indices, random_seed, max_iterations=100):
        """Try to improve by swapping blocks: remove one, add another if it helps."""
        if not chosen_indices:
            return chosen_indices
        
        rng = random.Random(random_seed)
        chosen_set = set(chosen_indices)
        current_size = len(chosen_indices)
        
        # Compute coverage counts
        cover_count = np.zeros(num_t_subsets, dtype=np.int32)
        for idx in chosen_indices:
            mask = block_masks[idx]
            m = mask
            while m:
                lsb = m & -m
                idx_bit = lsb.bit_length() - 1
                cover_count[idx_bit] += 1
                m ^= lsb
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            if time.time() > deadline:
                break
            improved = False
            iterations += 1
            
            # Try removing each block and see if we can add a better one
            for i in range(len(chosen_indices)):
                if time.time() > deadline:
                    break
                remove_idx = chosen_indices[i]
                remove_mask = block_masks[remove_idx]
                
                # Check if removing this block leaves any subset uncovered
                m = remove_mask
                can_remove = True
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    if cover_count[idx_bit] <= 1:
                        can_remove = False
                        break
                    m ^= lsb
                
                if not can_remove:
                    continue
                
                # Temporarily remove
                m = remove_mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    cover_count[idx_bit] -= 1
                    m ^= lsb
                
                # Find the best block to add that covers the most newly uncovered subsets
                uncovered_mask = 0
                m = remove_mask
                while m:
                    lsb = m & -m
                    idx_bit = lsb.bit_length() - 1
                    if cover_count[idx_bit] == 0:
                        uncovered_mask |= lsb
                    m ^= lsb
                
                if uncovered_mask == 0:
                    # No subsets became uncovered, so we can just remove
                    chosen_indices.pop(i)
                    chosen_set.discard(remove_idx)
                    improved = True
                    break
                
                # Find best block to add
                best_gain = -1
                best_add_idx = -1
                for j in range(num_blocks):
                    if j in chosen_set:
                        continue
                    intersection = block_masks[j] & uncovered_mask
                    gain = count_bits(intersection)
                    if gain > best_gain:
                        best_gain = gain
                        best_add_idx = j
                
                if best_add_idx != -1 and best_gain > 0:
                    # Add the best block
                    chosen_indices.append(best_add_idx)
                    chosen_set.add(best_add_idx)
                    m = block_masks[best_add_idx]
                    while m:
                        lsb = m & -m
                        idx_bit = lsb.bit_length() - 1
                        cover_count[idx_bit] += 1
                        m ^= lsb
                    improved = True
                else:
                    # Can't improve, restore the removed block
                    m = remove_mask
                    while m:
                        lsb = m & -m
                        idx_bit = lsb.bit_length() - 1
                        cover_count[idx_bit] += 1
                        m ^= lsb
                    chosen_indices.insert(i, remove_idx)
                    chosen_set.add(remove_idx)
        
        return chosen_indices

    # Run multiple restarts
    seed = 0
    while time.time() < deadline:
        indices = greedy_cover(seed)
        if indices is None:
            break
            
        reduced_indices = reduce_cover(indices)
        
        # Apply local search swap
        swapped_indices = local_search_swap(reduced_indices, seed, max_iterations=50)
        
        if len(swapped_indices) < best_size:
            best_size = len(swapped_indices)
            best_solution = [list(all_blocks[i]) for i in swapped_indices]
        elif len(reduced_indices) < best_size:
            best_size = len(reduced_indices)
            best_solution = [list(all_blocks[i]) for i in reduced_indices]
            
        seed += 1

    if best_solution is None:
        # Fallback: simple greedy without reduction
        indices = greedy_cover(0)
        if indices:
            best_solution = [list(all_blocks[i]) for i in indices]
        else:
            return []
            
    return best_solution
