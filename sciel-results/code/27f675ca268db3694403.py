import itertools
import time
import random
import sys

# Global Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = list(itertools.combinations(range(V), K))
N_BLOCKS = len(all_blocks)

all_masks = [0] * N_BLOCKS
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for i, block in enumerate(all_blocks):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(i)
    all_masks[i] = mask

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
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

    best_len = float('inf')
    best_blocks = []

    # We can try a few restarts if time permits
    # But usually, one good run with strong local search is better.
    # Let's stick to one robust run but enhance the local search.

    # 1. Greedy Initialization
    # We use a randomized greedy to avoid deterministic traps
    current_blocks = []
    current_block_set = set()
    counts = [0] * N_SUBSETS
    
    # Precompute mask for each block as a set of indices for faster intersection?
    # Bitmasks are efficient for intersection and counting.
    # N_SUBSETS = 3876. Python integers can handle this.

    def add_block(idx):
        current_blocks.append(idx)
        current_block_set.add(idx)
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(idx):
        current_block_set.remove(idx)
        pos = current_blocks.index(idx)
        current_blocks[pos] = current_blocks[-1]
        current_blocks.pop()
        
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb

    def is_redundant(idx):
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    # Greedy Loop
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        if time.time() > deadline - 60:
            break
        best_idx = -1
        best_gain = 0
        best_overlap = float('inf') # Tie breaker: minimize overlap with existing blocks
        
        # To speed up, we can iterate. 77520 blocks is manageable.
        for i in range(N_BLOCKS):
            if i in current_block_set:
                continue
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Calculate overlap for tie-breaking
                overlap = (all_masks[i] & ~uncovered_mask).bit_count()
                best_overlap = overlap
            elif gain == best_gain and gain > 0:
                overlap = (all_masks[i] & ~uncovered_mask).bit_count()
                if overlap < best_overlap:
                    best_idx = i
                    best_overlap = overlap

        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    # Initial solution found
    # Prune initial solution
    for i in range(len(current_blocks) - 1, -1, -1):
        if is_redundant(current_blocks[i]):
            remove_block(current_blocks[i])
            
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks = current_blocks.copy()

    # 2. Local Search
    # Enhanced with 2-for-1 swaps
    
    no_improve_count = 0
    max_no_improve = 100
    current_best_len = len(current_blocks)
    
    while time.time() < deadline - 5:
        # Phase 1: Prune redundant blocks
        removed_any = False
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                break
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                break
        
        if removed_any:
            no_improve_count = 0
            continue

        # Phase 2: Try 1-for-1 swap
        found_swap = False
        if current_blocks:
            i = random.randint(0, len(current_blocks) - 1)
            idx = current_blocks[i]
            
            # Find unique subsets covered by idx
            m = all_masks[idx]
            unique_bits = []
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_bits.append(bit_idx)
                mm ^= lsb
                
            if not unique_bits:
                # It was redundant, but pruning missed it?
                remove_block(idx)
                no_improve_count = 0
                continue
            else:
                # Limit search if too many unique bits
                if len(unique_bits) <= 4:
                    unique_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                    
                    candidates = set(blocks_containing_subset[unique_bits[0]])
                    for b in unique_bits[1:]:
                        if not candidates:
                            break
                        candidates &= set(blocks_containing_subset[b])
                    
                    valid_candidates = [c for c in candidates if c not in current_block_set]
                    
                    if valid_candidates:
                        new_idx = random.choice(valid_candidates)
                        remove_block(idx)
                        add_block(new_idx)
                        found_swap = True
                        no_improve_count = 0

        # Phase 3: Try 2-for-1 swap
        if not found_swap and len(current_blocks) >= 2:
            # Pick two random blocks
            i1 = random.randint(0, len(current_blocks) - 1)
            i2 = random.randint(0, len(current_blocks) - 1)
            if i1 == i2:
                i2 = (i2 + 1) % len(current_blocks)
            
            idx1 = current_blocks[i1]
            idx2 = current_blocks[i2]
            
            # Find union of unique subsets for both
            # A subset is unique to {idx1, idx2} if count is 1 or 2 and it is in idx1 or idx2
            # Actually, we need to find subsets that are ONLY covered by idx1 or idx2.
            # If a subset is covered by idx1 and idx3, removing idx1 is fine.
            # We want to remove idx1 and idx2.
            # Subsets that will become uncovered are those where count == 1 and in idx1 or idx2.
            # Subsets where count == 2 and in both idx1 and idx2 will become uncovered if count was 2?
            # No, if count is 2 and it's in idx1 and idx2, removing both makes count 0.
            # So we need to cover:
            # 1. Subsets in idx1 with count == 1
            # 2. Subsets in idx2 with count == 1
            # 3. Subsets in idx1 AND idx2 with count == 2
            
            m1 = all_masks[idx1]
            m2 = all_masks[idx2]
            
            unique_to_1_or_2 = []
            
            # Check subsets in m1
            mm = m1
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_to_1_or_2.append(bit_idx)
                elif counts[bit_idx] == 2 and (m2 >> bit_idx) & 1:
                    unique_to_1_or_2.append(bit_idx)
                mm ^= lsb
                
            # Check subsets in m2 that are not in m1 (to avoid duplicates)
            mm = m2 & ~m1
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_to_1_or_2.append(bit_idx)
                mm ^= lsb

            if unique_to_1_or_2 and len(unique_to_1_or_2) <= 6:
                unique_to_1_or_2.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                candidates = set(blocks_containing_subset[unique_to_1_or_2[0]])
                for b in unique_to_1_or_2[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                
                valid_candidates = [c for c in candidates if c not in current_block_set]
                
                if valid_candidates:
                    new_idx = random.choice(valid_candidates)
                    # Remove both, add one
                    remove_block(idx1)
                    remove_block(idx2)
                    add_block(new_idx)
                    found_swap = True
                    no_improve_count = 0

        if not found_swap:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                n_remove = random.randint(3, 6)
                to_remove = random.sample(current_blocks, min(n_remove, len(current_blocks)))
                for idx in to_remove:
                    remove_block(idx)
                
                # Re-greedy
                if time.time() < deadline - 10:
                    uncov_mask = 0
                    for s in range(N_SUBSETS):
                        if counts[s] == 0:
                            uncov_mask |= 1 << s
                            
                    while uncov_mask:
                        if time.time() > deadline - 5:
                            break
                        best_idx = -1
                        best_gain = 0
                        best_overlap = float('inf')
                        for i in range(N_BLOCKS):
                            if i in current_block_set:
                                continue
                            gain = (all_masks[i] & uncov_mask).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = i
                                overlap = (all_masks[i] & ~uncov_mask).bit_count()
                                best_overlap = overlap
                            elif gain == best_gain and gain > 0:
                                overlap = (all_masks[i] & ~uncov_mask).bit_count()
                                if overlap < best_overlap:
                                    best_idx = i
                                    best_overlap = overlap
                        if best_idx == -1:
                            break
                        add_block(best_idx)
                        uncov_mask &= ~all_masks[best_idx]
                        
                no_improve_count = 0
                max_no_improve = min(200, max_no_improve * 2)
                # Update best
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()

    # Final cleanup
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(current_blocks) - 1, -1, -1):
            idx = current_blocks[i]
            if is_redundant(idx):
                remove_block(idx)
                changed = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()

    return [list(all_blocks[i]) for i in best_blocks]
