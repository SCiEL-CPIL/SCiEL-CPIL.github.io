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
# blocks_containing_subset is a list of lists for fallback or debugging, 
# but we will use bitsets for speed in the main loop.
# SubsetBlockMasks[s] is a big int where bit i is 1 if all_blocks[i] contains subsets[s]
SubsetBlockMasks = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    idx = len(all_blocks)
    mask = 0
    for sub in itertools.combinations(block, T):
        s_idx = subset_to_idx[sub]
        mask |= 1 << s_idx
        SubsetBlockMasks[s_idx] |= 1 << idx
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)
ALL_BLOCKS_MASK = (1 << N_BLOCKS) - 1

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
    deadline = start_time + 115  # 5s margin
    
    # 1. Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    current_block_mask = 0  # Bitmask of which block indices are in current_blocks

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
        current_block_mask |= (1 << idx)
        
    def remove_block(idx):
        # Find position in list? 
        # For efficiency in ILS, we might want to swap-remove or just pop.
        # Since we track indices, we can just remove from list.
        # But current_block_mask is a bitmask, so we can clear bit.
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        
        # Remove from list
        # This is O(N). N is ~150. Acceptable.
        pos = current_blocks.index(idx)
        current_blocks.pop(pos)
        current_masks.pop(pos)
        current_block_mask &= ~(1 << idx)

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

    # Greedy
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Find block with max gain
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
    best_block_mask = current_block_mask
    
    # 2. Local Search with Perturbation
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Phase 1: Prune
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
                    best_block_mask = current_block_mask
                # Restart pruning loop
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # Phase 2: Swap
        if len(current_blocks) == 0:
            break
            
        i = random.randint(0, len(current_blocks) - 1)
        
        # Tentatively remove i
        m_i = all_masks[i]
        # Save counts? No, just decrement and check.
        # We need to identify critical bits.
        critical_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                critical_bits.append(bit_idx)
            mm ^= lsb
            
        if not critical_bits:
            # Redundant, remove permanently
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
                best_block_mask = current_block_mask
            no_improve_count = 0
            continue
            
        # If critical bits exist, we need to find a replacement block j
        # that covers all critical bits.
        # j must not be in current_blocks (excluding i? No, i is being removed, so j != i.
        # Also j should not be in current_blocks to be a "new" block?
        # If j is in current_blocks, it's already there.
        # But if j is in current_blocks, did it cover the critical bits?
        # If j covered a critical bit s, then counts[s] would have been >= 2 (from i and j).
        # After removing i, counts[s] would be 1.
        # But we defined critical_bits based on counts BEFORE removing i?
        # No, my `is_redundant` and `critical_bits` logic above uses `counts` which includes i.
        # So if counts[s] == 1, only i covers s.
        # So NO other block in current_blocks covers s.
        # Thus, j MUST be a block NOT in current_blocks.
        
        # Find candidate j
        cand_mask = ALL_BLOCKS_MASK
        # Intersect with masks of blocks containing each critical bit
        for s in critical_bits:
            cand_mask &= SubsetBlockMasks[s]
        
        # Exclude blocks already in cover
        cand_mask &= ~current_block_mask
        
        if cand_mask:
            # Pick a random candidate
            # Get a random set bit
            # Trick: mask with random bits? Or just iterate?
            # Since cand_mask is a big int, we can pick a random bit.
            # Simple: convert to list of indices? Too slow if many.
            # Better: pick a random integer, and find a set bit in cand_mask & (1 << random_idx)?
            # Or: 
            # Let's just pick the first set bit for simplicity, or a random one.
            # To be random:
            # We can use: 
            # num_bits = cand_mask.bit_count()
            # target = random.randint(0, num_bits - 1)
            # Then find the (target+1)-th set bit.
            # This is O(N_BLOCKS) in worst case.
            # Given N_BLOCKS=116k, and this happens often, it might be slow.
            # Alternative: Just pick the lowest set bit. It's fast.
            lsb = cand_mask & -cand_mask
            j = lsb.bit_length() - 1
            
            # Perform swap
            remove_block(i)
            add_block(j)
            no_improve_count = 0
        else:
            # No swap possible
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation
                # Add a random block not in cover
                avail_mask = ALL_BLOCKS_MASK & ~current_block_mask
                if avail_mask:
                    lsb = avail_mask & -avail_mask
                    c = lsb.bit_length() - 1
                    add_block(c)
                    no_improve_count = 0
                    max_no_improve = min(200, max_no_improve * 2)
                else:
                    break

    return [list(all_blocks[i]) for i in best_blocks]
