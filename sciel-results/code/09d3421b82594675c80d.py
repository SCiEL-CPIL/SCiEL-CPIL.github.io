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
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute weights for weighted greedy
# Weight is inverse of frequency, normalized
max_freq = max(subset_frequencies)
weights = [1.0 / f for f in subset_frequencies]

# Precompute bit positions for each mask for faster iteration
# This is a list of lists of bit indices
mask_bit_positions = []
for m in all_masks:
    positions = []
    mm = m
    while mm:
        lsb = mm & -mm
        bit_idx = lsb.bit_length() - 1
        positions.append(bit_idx)
        mm ^= lsb
    mask_bit_positions.append(positions)

# Precompute for each block, which other blocks share subsets with it
# This can help in local search
# But for now, let's focus on the main algorithm

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
    deadline = start_time + 118  # Safety margin
    
    # 1. Weighted Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        for bit_idx in mask_bit_positions[idx]:
            counts[bit_idx] += 1

    def remove_block(i):
        m = current_masks[i]
        for bit_idx in mask_bit_positions[i]:
            counts[bit_idx] -= 1
        # Remove from lists
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        for bit_idx in mask_bit_positions[i]:
            if counts[bit_idx] == 1:
                return False
        return True

    # Weighted Greedy with improved gain computation
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0.0
        for i in range(N_BLOCKS):
            m = all_masks[i]
            gain = 0.0
            mm = m & uncovered_mask
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                gain += weights[bit_idx]
                mm ^= lsb
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_gain == 0:
            break # Should not happen if uncovered_mask is not 0
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 150
    
    def try_remove_and_swap():
        """Try to remove one block and add one or two blocks to maintain coverage.
        Returns True if a swap was made."""
        nonlocal no_improve_count, best_len, best_blocks
        
        if len(current_blocks) == 0:
            return False
        
        # Pick a random block to remove - prefer blocks with higher redundancy
        n_blocks = len(current_blocks)
        # Weight selection: blocks with more shared coverage are better candidates
        redundancies = []
        for i in range(n_blocks):
            red = 0
            for bit_idx in mask_bit_positions[i]:
                if counts[bit_idx] > 1:
                    red += 1
            redundancies.append(red)
        
        # Weighted random selection
        total_red = sum(redundancies)
        if total_red > 0:
            r = random.random() * total_red
            cum = 0
            i = 0
            for idx in range(n_blocks):
                cum += redundancies[idx]
                if r <= cum:
                    i = idx
                    break
        else:
            i = random.randint(0, n_blocks - 1)
        
        # Find uncovered subsets if we remove block i
        uncovered_bits = []
        for bit_idx in mask_bit_positions[i]:
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            
        if not uncovered_bits:
            # Block is redundant, just remove it
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            return True
        
        # Try 1-exchange: find a block covering all uncovered_bits
        if len(uncovered_bits) <= 6:
            uncovered_sorted = sorted(uncovered_bits, key=lambda b: len(blocks_containing_subset[b]))
            candidates = list(blocks_containing_subset[uncovered_sorted[0]])
            
            for b in uncovered_sorted[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
            
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Choose candidate that covers most other uncovered subsets
                    best_cand = max(valid_candidates, key=lambda c: sum(1 for bit_idx in uncovered_bits 
                                                                       if all_masks[c] & (1 << bit_idx)))
                    remove_block(i)
                    add_block(best_cand)
                    return True
        
        # Try 2-exchange: remove i, add two blocks that together cover uncovered_bits
        # Strategy: for each uncovered bit, find blocks that cover it
        # Then find pairs that cover all uncovered bits
        if len(uncovered_bits) <= 8:
            # Build candidate pool: blocks that cover at least one uncovered bit
            # and are not already in current_blocks
            current_block_set = set(current_blocks)
            candidate_pool = set()
            for bit_idx in uncovered_bits:
                for b in blocks_containing_subset[bit_idx]:
                    if b != i and b not in current_block_set:
                        candidate_pool.add(b)
            
            if candidate_pool:
                # Sort candidates by how many uncovered bits they cover (descending)
                cand_list = list(candidate_pool)
                cand_list.sort(key=lambda b: sum(1 for bit_idx in uncovered_bits 
                                               if all_masks[b] & (1 << bit_idx)), reverse=True)
                
                # Limit search for speed
                max_cands = min(len(cand_list), 30)
                cand_list = cand_list[:max_cands]
                
                # Try pairs from the top candidates
                found = False
                for idx1 in range(len(cand_list)):
                    if time.time() > deadline:
                        return False
                    j1 = cand_list[idx1]
                    mask_j1 = all_masks[j1]
                    
                    # Check which uncovered bits j1 covers
                    remaining = [b for b in uncovered_bits if not (mask_j1 & (1 << b))]
                    
                    if not remaining:
                        # j1 covers all uncovered bits, but we already tried 1-exchange
                        # This shouldn't happen, but just in case
                        remove_block(i)
                        add_block(j1)
                        return True
                    
                    # Find j2 that covers all remaining bits
                    for idx2 in range(idx1 + 1, len(cand_list)):
                        j2 = cand_list[idx2]
                        mask_j2 = all_masks[j2]
                        
                        covers_remaining = True
                        for bit_idx in remaining:
                            if not (mask_j2 & (1 << bit_idx)):
                                covers_remaining = False
                                break
                        
                        if covers_remaining:
                            remove_block(i)
                            add_block(j1)
                            add_block(j2)
                            return True
        
        return False
    
    def polish_pass():
        """Final polish: repeatedly remove redundant blocks until none remain."""
        nonlocal best_len, best_blocks
        
        while True:
            if time.time() > deadline:
                break
            removed = False
            for i in range(len(current_blocks)):
                if is_redundant(i):
                    remove_block(i)
                    removed = True
                    if len(current_blocks) < best_len:
                        best_len = len(current_blocks)
                        best_blocks = current_blocks.copy()
                    break
            if not removed:
                break
    
    while time.time() < deadline:
        # First, try to remove redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            continue
        
        # Try a swap: remove one block, add one or two
        if try_remove_and_swap():
            no_improve_count = 0
            continue
        
        # No improvement, increment counter
        no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Perturbation: Add a random block, then try to prune
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
            no_improve_count = 0
    
    # Final polish pass
    polish_pass()
    
    return [list(all_blocks[i]) for i in best_blocks]
