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

# Precompute numpy masks for faster gain computation
# Each block covers C(7,4)=35 subsets
# We'll store as numpy array of shape (N_BLOCKS, N_SUBSETS) with 0/1 values
# But N_BLOCKS * N_SUBSETS might be large, so use sparse-like approach
# Actually N_BLOCKS = C(19,7) = 50388, N_SUBSETS = C(19,4) = 3876
# 50388 * 3876 = ~195M entries, too large for dense numpy
# So we'll use the bitmask approach but optimize the gain computation

# Precompute for each block, the list of subset indices it covers
block_subset_indices = []
for mask in all_masks:
    idxs = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bit_idx = lsb.bit_length() - 1
        idxs.append(bit_idx)
        mm ^= lsb
    block_subset_indices.append(idxs)

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
    
    # 1. Weighted Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        # Remove from lists
        current_blocks.pop(i)
        current_masks.pop(i)

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

    # Weighted Greedy with optimized gain computation
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0.0
        for i in range(N_BLOCKS):
            # Use precomputed subset indices for faster gain calculation
            gain = 0.0
            for bit_idx in block_subset_indices[i]:
                if uncovered_mask & (1 << bit_idx):
                    gain += weights[bit_idx]
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
    max_no_improve = 500
    current_block_set = set(current_blocks)  # For O(1) membership testing
    
    def add_block_optimized(idx):
        current_blocks.append(idx)
        current_block_set.add(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block_optimized(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        current_block_set.discard(current_blocks[i])
        current_blocks.pop(i)
        current_masks.pop(i)

    def greedy_restore(max_blocks=None):
        """Add blocks greedily until all subsets are covered. Returns number added."""
        if max_blocks is None:
            max_blocks = 20
        added = 0
        for _ in range(max_blocks):
            if time.time() > deadline:
                break
            # Find uncovered subsets
            best_j = -1
            best_gain = 0
            for j in range(N_BLOCKS):
                if j in current_block_set:
                    continue
                gain = 0
                for bit_idx in block_subset_indices[j]:
                    if counts[bit_idx] == 0:
                        gain += 1
                        if gain <= best_gain:
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
                    if best_gain >= 30:
                        break
            if best_j < 0 or best_gain == 0:
                break
            add_block_optimized(best_j)
            added += 1
        return added

    while time.time() < deadline:
        # Try to remove redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline: break
            if is_redundant(i):
                remove_block_optimized(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # If no redundant block found, try a swap
        if len(current_blocks) == 0: break
        
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Tentatively remove
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
            remove_block_optimized(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        # We need to cover uncovered_bits
        # Try 1-exchange: find a block covering all uncovered_bits
        success = False
        if len(uncovered_bits) <= 5:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates: break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                valid_candidates = [c for c in candidates if c != i and c not in current_block_set]
                if not valid_candidates:
                    valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    # Score each candidate by how many NEW subsets it covers
                    # (subsets currently covered by exactly 1 block, i.e., the removed block)
                    best_j = None
                    best_score = -1
                    for j in valid_candidates[:30]:
                        score = 0
                        for bit_idx in block_subset_indices[j]:
                            if counts[bit_idx] == 1:
                                score += 1
                        if score > best_score:
                            best_score = score
                            best_j = j
                    if best_j is not None:
                        remove_block_optimized(i)
                        add_block_optimized(best_j)
                        no_improve_count = 0
                        success = True
        
        if success:
            continue

        # Try 2-exchange: remove i, add two blocks
        if not success and no_improve_count > 30:
            for _ in range(15):
                if time.time() > deadline: break
                j1 = random.randint(0, N_BLOCKS - 1)
                j2 = random.randint(0, N_BLOCKS - 1)
                if j1 == j2 or j1 == i or j2 == i:
                    continue
                if j1 in current_block_set or j2 in current_block_set:
                    continue
                
                combined_mask = all_masks[j1] | all_masks[j2]
                covers_all = True
                for bit_idx in uncovered_bits:
                    if not (combined_mask & (1 << bit_idx)):
                        covers_all = False
                        break
                
                if covers_all:
                    remove_block_optimized(i)
                    add_block_optimized(j1)
                    add_block_optimized(j2)
                    no_improve_count = 0
                    success = True
                    break
        
        if not success:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: remove a random block, then greedily restore
                i2 = random.randint(0, len(current_blocks) - 1)
                remove_block_optimized(i2)
                greedy_restore(max_blocks=5)
                no_improve_count = 0

    # Final cleanup: restore best solution and try to remove more blocks
    current_blocks = best_blocks.copy()
    current_block_set = set(current_blocks)
    current_masks = [all_masks[i] for i in current_blocks]
    counts = [0] * N_SUBSETS
    for i in range(len(current_blocks)):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb
    
    # Try to remove redundant blocks from best solution
    for _ in range(100):
        if time.time() > deadline: break
        removed = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        for i in indices:
            if is_redundant(i):
                remove_block_optimized(i)
                removed = True
                break
        if not removed:
            break
    
    return [list(all_blocks[i]) for i in current_blocks]
