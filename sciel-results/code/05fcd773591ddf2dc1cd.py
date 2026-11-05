import itertools
import random
import time
import math
import numpy as np

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute for faster gain calculation
# For each block, store the list of subset indices it covers
block_subsets = []
for mask in all_masks:
    subs = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bit_idx = lsb.bit_length() - 1
        subs.append(bit_idx)
        mm ^= lsb
    block_subsets.append(subs)

# Precompute subset frequencies (how many blocks cover each subset)
subset_freq = [0] * N_SUBSETS
for subs in block_subsets:
    for s in subs:
        subset_freq[s] += 1

# Precompute block "value" based on rarity of subsets it covers
block_value = [0.0] * N_BLOCKS
for i, subs in enumerate(block_subsets):
    for s in subs:
        block_value[i] += 1.0 / subset_freq[s]

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def greedy_init(seed):
        """Greedy initialization with weighted gain based on subset rarity."""
        rng = random.Random(seed)
        counts = [0] * N_SUBSETS
        current_blocks = []
        current_masks = []
        covered_mask = 0
        
        # Track which blocks are already selected
        selected = set()
        
        while True:
            # Check if all subsets are covered
            uncovered_count = 0
            for s in range(N_SUBSETS):
                if counts[s] == 0:
                    uncovered_count += 1
                    if uncovered_count > 0:
                        break
            
            if uncovered_count == 0:
                break
            
            # Find best block: maximize weighted gain
            # Weight = 1/freq[s] for each uncovered subset s in block
            best_idx = -1
            best_gain = -1.0
            
            # Use a random subset of blocks to speed up, or all if time allows
            if len(current_blocks) < 20:
                candidates = range(N_BLOCKS)
            else:
                # Sample blocks to speed up
                sample_size = min(5000, N_BLOCKS)
                candidates = rng.sample(range(N_BLOCKS), sample_size)
            
            for i in candidates:
                if i in selected:
                    continue
                
                # Compute weighted gain for this block
                gain = 0.0
                m = all_masks[i]
                mm = m
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] == 0:
                        gain += 1.0 / subset_freq[bit_idx]
                    mm ^= lsb
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            
            if best_idx == -1 or best_gain <= 0:
                # No improvement possible, break
                break
            
            # Add the block
            selected.add(best_idx)
            current_blocks.append(best_idx)
            current_masks.append(all_masks[best_idx])
            m = all_masks[best_idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                mm ^= lsb
            
            # Early exit if time is running out
            if time.time() > deadline - 10:
                break
        
        return current_blocks, counts, current_masks
    
    def local_search(initial_blocks, initial_counts, initial_masks, seed):
        """Local search: remove redundant blocks, then try swaps."""
        rng = random.Random(seed)
        current_blocks = initial_blocks[:]
        current_masks = initial_masks[:]
        counts = initial_counts[:]
        
        best_len = len(current_blocks)
        best_blocks = current_blocks[:]
        
        no_improve_count = 0
        max_no_improve = 30
        
        def is_redundant(i):
            m = current_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True
        
        def try_remove_redundant():
            nonlocal best_len, best_blocks, no_improve_count
            indices = list(range(len(current_blocks)))
            rng.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    break
                if is_redundant(i):
                    # Remove block i
                    m = current_masks[i]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        mm ^= lsb
                    current_blocks.pop(i)
                    current_masks.pop(i)
                    
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks[:]
                    no_improve_count = 0
                    return True
            return False
        
        while time.time() < deadline - 5:
            # Try to remove redundant blocks
            if try_remove_redundant():
                continue
            
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Perturbation: add a random block, then try to prune
                c = rng.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    current_blocks.append(c)
                    current_masks.append(all_masks[c])
                    m = all_masks[c]
                    mm = m
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] += 1
                        mm ^= lsb
                no_improve_count = 0
                continue
            
            # Try a swap: remove one block, add another
            if len(current_blocks) == 0:
                break
            
            i = rng.randint(0, len(current_blocks) - 1)
            
            # Tentatively remove block i
            m_i = current_masks[i]
            uncovered_bits = []
            mm = m_i
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    uncovered_bits.append(bit_idx)
                mm ^= lsb
            
            if not uncovered_bits:
                # Block i is redundant, just remove it
                mm = m_i
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    counts[bit_idx] -= 1
                    mm ^= lsb
                current_blocks.pop(i)
                current_masks.pop(i)
                
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks[:]
                no_improve_count = 0
                continue
            
            # Find a replacement block that covers all uncovered_bits
            if len(uncovered_bits) > 4:
                # Too many uncovered subsets, skip this swap
                continue
            
            # Find candidate blocks covering all uncovered_bits
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = list(blocks_containing_subset[uncovered_bits[0]])
            
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
            
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Choose the candidate that maximizes additional coverage
                    best_j = valid_candidates[0]
                    best_additional = -1
                    
                    # Limit search for speed
                    search_limit = min(50, len(valid_candidates))
                    for j in valid_candidates[:search_limit]:
                        additional = (all_masks[j] & ~m_i).bit_count()
                        if additional > best_additional:
                            best_additional = additional
                            best_j = j
                    
                    # Perform swap
                    # Remove i
                    mm = m_i
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] -= 1
                        mm ^= lsb
                    current_blocks.pop(i)
                    current_masks.pop(i)
                    
                    # Add best_j
                    current_blocks.append(best_j)
                    current_masks.append(all_masks[best_j])
                    m_j = all_masks[best_j]
                    mm = m_j
                    while mm:
                        lsb = mm & -mm
                        bit_idx = lsb.bit_length() - 1
                        counts[bit_idx] += 1
                        mm ^= lsb
                    
                    no_improve_count = 0
                    continue
        
        return best_blocks, best_len
    
    # Run multiple greedy restarts with different seeds
    best_overall_blocks = None
    best_overall_len = float('inf')
    
    # First greedy with seed 42
    init_blocks, init_counts, init_masks = greedy_init(42)
    if init_blocks:
        ls_blocks, ls_len = local_search(init_blocks, init_counts, init_masks, 42)
        if ls_len < best_overall_len:
            best_overall_len = ls_len
            best_overall_blocks = ls_blocks[:]
    
    # Second greedy with seed 123
    if time.time() < deadline - 20:
        init_blocks2, init_counts2, init_masks2 = greedy_init(123)
        if init_blocks2:
            ls_blocks2, ls_len2 = local_search(init_blocks2, init_counts2, init_masks2, 123)
            if ls_len2 < best_overall_len:
                best_overall_len = ls_len2
                best_overall_blocks = ls_blocks2[:]
    
    # Third greedy with seed 456 if time permits
    if time.time() < deadline - 15:
        init_blocks3, init_counts3, init_masks3 = greedy_init(456)
        if init_blocks3:
            ls_blocks3, ls_len3 = local_search(init_blocks3, init_counts3, init_masks3, 456)
            if ls_len3 < best_overall_len:
                best_overall_len = ls_len3
                best_overall_blocks = ls_blocks3[:]
    
    if best_overall_blocks is None:
        # Fallback: just return the first greedy result
        best_overall_blocks = init_blocks
    
    # Final cleanup: remove any remaining redundant blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_overall_blocks:
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            final_counts[bit_idx] += 1
            mm ^= lsb
    
    final_blocks = best_overall_blocks[:]
    for i in range(len(final_blocks) - 1, -1, -1):
        m = all_masks[final_blocks[i]]
        mm = m
        redundant = True
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if final_counts[bit_idx] == 1:
                redundant = False
                break
            mm ^= lsb
        if redundant:
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                final_counts[bit_idx] -= 1
                mm ^= lsb
            final_blocks.pop(i)
    
    return [list(all_blocks[i]) for i in final_blocks]
