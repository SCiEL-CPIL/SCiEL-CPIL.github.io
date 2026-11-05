import itertools
import time
import random

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

    # 1. Greedy Initialization
    uncovered_mask = (1 << N_SUBSETS) - 1
    current_blocks = []
    current_block_set = set()
    counts = [0] * N_SUBSETS
    
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
        # Remove from list
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
    while uncovered_mask:
        if time.time() > deadline - 60:
            break
        best_idx = -1
        best_gain = 0
        # Scan all blocks
        for i in range(N_BLOCKS):
            if i in current_block_set:
                continue
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        # Phase 1: Prune redundant blocks
        removed_any = False
        # Shuffle indices to randomize pruning order
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
            max_no_improve = max(50, max_no_improve // 2)
            continue
            
        # Phase 2: Try 1-for-1 swap
        if not current_blocks:
            break
            
        found_swap = False
        # Try a random block for swap
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
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
        else:
            # Limit search if too many unique bits
            if len(unique_bits) <= 4:
                # Sort by constraint tightness
                unique_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                
                # Intersection of candidate lists
                candidates = set(blocks_containing_subset[unique_bits[0]])
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    candidates &= set(blocks_containing_subset[b])
                
                # Filter out blocks already in cover
                valid_candidates = [c for c in candidates if c not in current_block_set]
                
                if valid_candidates:
                    new_idx = random.choice(valid_candidates)
                    remove_block(idx)
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
                    # Compute uncovered
                    uncov_mask = 0
                    for s in range(N_SUBSETS):
                        if counts[s] == 0:
                            uncov_mask |= 1 << s
                            
                    while uncov_mask:
                        if time.time() > deadline - 5:
                            break
                        best_idx = -1
                        best_gain = 0
                        for i in range(N_BLOCKS):
                            if i in current_block_set:
                                continue
                            gain = (all_masks[i] & uncov_mask).bit_count()
                            if gain > best_gain:
                                best_gain = gain
                                best_idx = i
                        if best_idx == -1:
                            break
                        add_block(best_idx)
                        uncov_mask &= ~all_masks[best_idx]
                        
                no_improve_count = 0
                max_no_improve = min(200, max_no_improve * 2)

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
