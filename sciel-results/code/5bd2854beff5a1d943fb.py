import itertools
import random
import time

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

# Precompute: for each block, list of subset indices it covers
block_subsets = []
for m in all_masks:
    subs = []
    mm = m
    while mm:
        lsb = mm & -mm
        subs.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_subsets.append(subs)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
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
    deadline = start_time + 145  # Safety margin for 150s kill
    
    # 1. Greedy Initialization - optimized
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        for s in block_subsets[idx]:
            counts[s] += 1

    def remove_block(i):
        for s in block_subsets[current_blocks[i]]:
            counts[s] -= 1
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        for s in block_subsets[current_blocks[i]]:
            if counts[s] == 1:
                return False
        return True

    # Greedy: pick block covering most uncovered subsets
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        for i in range(N_BLOCKS):
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
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 30
    
    while time.time() < deadline:
        # Phase 1: Prune redundant blocks
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
            
        if len(current_blocks) == 0:
            break
        
        # Phase 2: Try 1-for-1 swap
        i = random.randint(0, len(current_blocks) - 1)
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
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        if len(uncovered_bits) <= 4:
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            for b in uncovered_bits[1:]:
                if not candidates:
                    break
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
            
            if candidates:
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    remove_block(i)
                    add_block(j)
                    no_improve_count = 0
                    continue
        
        # Phase 3: Try 2-for-1 swap (remove 2 blocks, add 1)
        if len(current_blocks) >= 2 and random.random() < 0.3:
            i1 = random.randint(0, len(current_blocks) - 1)
            i2 = random.randint(0, len(current_blocks) - 1)
            if i1 == i2:
                i2 = (i2 + 1) % len(current_blocks)
            
            # Compute union of unique subsets covered by blocks i1 and i2
            m_union = current_masks[i1] | current_masks[i2]
            unique_bits = []
            mm = m_union
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                # Count how many of i1,i2 cover this bit
                c = 0
                if (current_masks[i1] >> bit_idx) & 1:
                    c += 1
                if (current_masks[i2] >> bit_idx) & 1:
                    c += 1
                total = counts[bit_idx]
                if total - c == 0:
                    unique_bits.append(bit_idx)
                mm ^= lsb
            
            if len(unique_bits) <= 3:
                unique_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
                candidates = blocks_containing_subset[unique_bits[0]]
                for b in unique_bits[1:]:
                    if not candidates:
                        break
                    set_cands = set(candidates)
                    set_b = set(blocks_containing_subset[b])
                    candidates = list(set_cands.intersection(set_b))
                
                if candidates:
                    valid_candidates = [c for c in candidates if c != i1 and c != i2]
                    if valid_candidates:
                        j = random.choice(valid_candidates)
                        # Remove both, add one
                        # Must remove in reverse order to preserve indices
                        if i1 > i2:
                            remove_block(i1)
                            remove_block(i2)
                        else:
                            remove_block(i2)
                            remove_block(i1)
                        add_block(j)
                        no_improve_count = 0
                        continue
        
        # Phase 4: Perturbation if stuck
        no_improve_count += 1
        if no_improve_count > max_no_improve:
            # Add a random block that covers some unique bits of a random block
            if current_blocks:
                rand_block_idx = random.randint(0, len(current_blocks) - 1)
                m_rb = current_masks[rand_block_idx]
                # Find bits with low count in this block's coverage
                low_count_bits = []
                mm = m_rb
                while mm:
                    lsb = mm & -mm
                    bit_idx = lsb.bit_length() - 1
                    if counts[bit_idx] <= 2:
                        low_count_bits.append(bit_idx)
                    mm ^= lsb
                
                if low_count_bits:
                    target_bit = random.choice(low_count_bits)
                    cands = blocks_containing_subset[target_bit]
                    if cands:
                        c = random.choice(cands)
                        if c not in current_blocks:
                            add_block(c)
                            no_improve_count = 0
                            max_no_improve = min(100, max_no_improve * 2)
                    continue
            else:
                c = random.randint(0, N_BLOCKS - 1)
                add_block(c)
                no_improve_count = 0
                max_no_improve = min(100, max_no_improve * 2)

    return [list(all_blocks[i]) for i in best_blocks]
